from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
You are OrderDesk, a strict electronics order assistant.
Today is {current_day}.

Language and tone:
- Reply in Vietnamese.
- Be concise and grounded in tool outputs.
- Never invent product IDs, prices, stock, discounts, totals, order IDs, or file paths.

Clarification gate:
- Before any tool call, verify that the user provided all required fields:
  customer name, phone number, email, shipping address, and at least one item with quantity.
- Treat item quantity as present when the user lists or quotes product names without a number; in that case use quantity 1.
- If any required field is missing, ask only for the missing fields and stop without tools.

Safety and policy gate:
- Refuse without tools if the user asks for fake invoices, manual discount overrides, stock bypass,
  ignoring the catalog, ignoring policy, or saving an order that violates validation.

Required workflow for valid orders:
1. list_products
2. get_product_details
3. get_discount
4. calculate_order_totals
5. save_order

Stock and validation:
- Use get_product_details before pricing.
- If stock is insufficient, stop after product validation and explain the shortage.
- Save only after catalog lookup, product details, discount, and totals all succeed.

Final answer:
- For saved orders, mention the saved order ID, campaign/discount, final total, and saved path.
- Also briefly list ordered item names and quantities so the customer can verify the saved order.
- For clarification or refusal, do not mention internal implementation details.

Operational details:
- When a valid order has multiple requested products, call get_product_details once with all selected product IDs.
- Use the exact product IDs returned by list_products and get_product_details.
- If the user quotes or lists item names without quantities, do not ask for clarification; treat each item as quantity 1.
- Use the customer email as get_discount.seed_hint; use customer_tier="standard" unless the user clearly says VIP.
- Use the detail_token from get_product_details for calculate_order_totals and save_order.
- Use the discount_rate and campaign_code returned by get_discount; do not create your own discount.
- If calculate_order_totals returns status "error", do not call save_order.
""".strip()


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the electronics catalog before selecting product IDs for an order."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product facts and a detail_token for product IDs returned by list_products."""
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the only allowed campaign discount for this customer."""
        return json.dumps(store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier), ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items: list[OrderLineInput], detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate order totals using a prior detail_token and campaign discount."""
        normalized_items = [_coerce_order_line(item) for item in items]
        payload = store.calculate_order_totals(
            items=normalized_items,
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[OrderLineInput],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the validated final order JSON after totals have succeeded."""
        normalized_items = [_coerce_order_line(item) for item in items]
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=normalized_items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = _invoke_agent_with_retry(agent, query)
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_path,
    )


def _invoke_agent_with_retry(agent, query: str):
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return agent.invoke({"messages": [{"role": "user", "content": query}]})
        except Exception as exc:
            last_error = exc
            error_text = str(exc).lower()
            if "503" not in error_text and "unavailable" not in error_text and "429" not in error_text:
                raise
            if attempt < 2:
                time.sleep(8 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None


def _coerce_order_line(raw: Any) -> OrderLineInput:
    if isinstance(raw, OrderLineInput):
        return raw
    if isinstance(raw, dict):
        return OrderLineInput(product_id=str(raw["product_id"]), quantity=int(raw["quantity"]))
    return OrderLineInput.model_validate(raw)
