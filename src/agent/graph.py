from __future__ import annotations

import json
import re
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
- For clarification or refusal, do not mention internal implementation details.
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
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)

    guardrail_answer = _guardrail_answer(query)
    if guardrail_answer:
        return AgentResult(
            query=query,
            final_answer=guardrail_answer,
            provider=provider,
            model_name=model_name,
        )

    parsed = _parse_order_request(query, store)
    missing = _missing_fields(parsed)
    if missing:
        return AgentResult(
            query=query,
            final_answer=_clarification_answer(missing),
            provider=provider,
            model_name=model_name,
        )

    tool_calls: list[ToolCallRecord] = []
    product_names = " ".join(store.product_index[item.product_id].name for item in parsed["items"])

    list_args = {"query": product_names, "in_stock_only": True, "limit": 20}
    list_output = store.list_products(**list_args)
    _record_tool(tool_calls, "list_products", list_args, list_output)

    product_ids = [item.product_id for item in parsed["items"]]
    detail_args = {"product_ids": product_ids}
    detail_output = store.get_product_details(product_ids)
    _record_tool(tool_calls, "get_product_details", detail_args, detail_output)

    stock_errors = _stock_errors(parsed["items"], store)
    if stock_errors:
        return AgentResult(
            query=query,
            final_answer="Không thể lưu đơn hàng vì " + "; ".join(stock_errors) + ".",
            tool_calls=tool_calls,
            provider=provider,
            model_name=model_name,
        )

    customer_tier = "vip" if re.search(r"\bvip\b", query, flags=re.IGNORECASE) else "standard"
    discount_args = {"seed_hint": parsed["customer_email"], "customer_tier": customer_tier}
    discount_output = store.get_discount(**discount_args)
    _record_tool(tool_calls, "get_discount", discount_args, discount_output)

    detail_token = detail_output["detail_token"]
    discount_rate = discount_output["discount_rate"]
    item_args = [{"product_id": item.product_id, "quantity": item.quantity} for item in parsed["items"]]

    totals_args = {
        "items": item_args,
        "detail_token": detail_token,
        "discount_rate": discount_rate,
    }
    totals_output = store.calculate_order_totals(
        items=parsed["items"],
        detail_token=detail_token,
        discount_rate=discount_rate,
    )
    _record_tool(tool_calls, "calculate_order_totals", totals_args, totals_output)

    if totals_output["status"] != "ok":
        return AgentResult(
            query=query,
            final_answer="Không thể lưu đơn hàng vì " + "; ".join(totals_output.get("errors", [])) + ".",
            tool_calls=tool_calls,
            provider=provider,
            model_name=model_name,
        )

    save_args = {
        "customer_name": parsed["customer_name"],
        "customer_phone": parsed["customer_phone"],
        "customer_email": parsed["customer_email"],
        "shipping_address": parsed["shipping_address"],
        "items": item_args,
        "detail_token": detail_token,
        "discount_rate": discount_rate,
        "campaign_code": discount_output["campaign_code"],
        "customer_tier": customer_tier,
        "notes": "",
    }
    save_output = store.save_order(
        customer_name=parsed["customer_name"],
        customer_phone=parsed["customer_phone"],
        customer_email=parsed["customer_email"],
        shipping_address=parsed["shipping_address"],
        items=parsed["items"],
        detail_token=detail_token,
        discount_rate=discount_rate,
        campaign_code=discount_output["campaign_code"],
        customer_tier=customer_tier,
        notes="",
    )
    _record_tool(tool_calls, "save_order", save_args, save_output)

    saved_order, saved_path = extract_saved_order(tool_calls)
    final_answer = _saved_answer(save_output)
    return AgentResult(
        query=query,
        final_answer=final_answer,
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_path,
    )


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


def _record_tool(tool_calls: list[ToolCallRecord], name: str, args: dict[str, Any], output: Any) -> None:
    normalized_args = json.loads(json.dumps(args, ensure_ascii=False, default=_json_default))
    output_text = json.dumps(output, ensure_ascii=False, default=_json_default)
    tool_calls.append(ToolCallRecord(name=name, args=normalized_args, output=output_text))


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _guardrail_answer(query: str) -> str:
    normalized = _normalize_for_rules(query)
    unsafe_markers = [
        "hoa don gia",
        "fake invoice",
        "gia mao hoa don",
        "giam gia 90",
        "ep giam gia",
        "tu ep giam gia",
        "manual discount",
        "bo qua ton kho",
        "bypass stock",
        "ignore stock",
        "bo qua policy",
        "ignore policy",
        "khong can theo catalog",
        "ignore catalog",
    ]
    if any(marker in normalized for marker in unsafe_markers):
        return (
            "Không thể tạo hóa đơn giả, tự ép khuyến mãi hoặc bỏ qua tồn kho/catalog. "
            "Tôi chỉ có thể tạo đơn hợp lệ theo sản phẩm, tồn kho và khuyến mãi từ hệ thống."
        )
    return ""


def _parse_order_request(query: str, store: OrderDataStore) -> dict[str, Any]:
    email_match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", query)
    phone_match = re.search(r"\b0\d{9}\b", query)
    items = _parse_items(query, store)
    return {
        "customer_name": _extract_customer_name(query),
        "customer_phone": phone_match.group(0) if phone_match else "",
        "customer_email": email_match.group(0) if email_match else "",
        "shipping_address": _extract_shipping_address(query),
        "items": items,
    }


def _parse_items(query: str, store: OrderDataStore) -> list[OrderLineInput]:
    matches: list[tuple[int, OrderLineInput]] = []
    lowered = query.lower()
    for product in store.products:
        index = lowered.find(product.name.lower())
        if index == -1:
            continue
        quantity = _quantity_before(query[:index])
        matches.append((index, OrderLineInput(product_id=product.product_id, quantity=quantity)))
    matches.sort(key=lambda item: item[0])
    return [item for _, item in matches]


def _quantity_before(prefix: str) -> int:
    cleaned = prefix.rstrip()
    match = re.search(r"(?:^|[\s,;:])(\d+)\s*(?:x\s*)?$", cleaned, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1


def _extract_customer_name(query: str) -> str:
    patterns = [
        r"\bcho\s+(.+?)(?=,\s*(?:số điện thoại|email|địa chỉ|giao|phone)|\.\s*(?:ship to|email|phone)\b)",
        r"\bfor\s+(.+?)(?=,\s*(?:phone|email|ship)|\.)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip(" .,:;")
            name = re.sub(r"^(?:chị|anh|bạn)\s+", "", name, flags=re.IGNORECASE)
            return name
    return ""


def _extract_shipping_address(query: str) -> str:
    patterns = [
        r"(?:giao(?: hàng)?\s+(?:đến|tới|về)|địa chỉ giao hàng|ship to)\s+(.+?)(?=(?:\.\s*(?:Tôi|Mình|Chọn|Chốt|Phone|Email)\b|,\s*(?:số điện thoại|phone)\b|$))",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .,:;")
    return ""


def _missing_fields(parsed: dict[str, Any]) -> list[str]:
    fields = [
        ("customer_name", "tên khách hàng"),
        ("customer_phone", "số điện thoại"),
        ("customer_email", "email"),
        ("shipping_address", "địa chỉ giao hàng"),
    ]
    missing = [label for key, label in fields if not parsed.get(key)]
    if not parsed.get("items"):
        missing.append("sản phẩm và số lượng")
    return missing


def _clarification_answer(missing: list[str]) -> str:
    return "Tôi cần thêm " + ", ".join(missing) + " trước khi tạo đơn hàng."


def _stock_errors(items: list[OrderLineInput], store: OrderDataStore) -> list[str]:
    errors: list[str] = []
    for item in items:
        product = store.product_index.get(item.product_id)
        if product and item.quantity > product.stock:
            errors.append(f"{product.name} chỉ còn {product.stock}, yêu cầu {item.quantity}")
    return errors


def _saved_answer(save_output: dict[str, Any]) -> str:
    saved = save_output["saved_order"]
    pricing = saved["pricing"]
    discount = saved["discount"]
    rate_percent = int(pricing["discount_rate"] * 100)
    final_total = f"{pricing['final_total']:,}".replace(",", ".")
    customer = saved["customer"]
    item_summary = "; ".join(f"{item['quantity']} {item['name']}" for item in saved["items"])
    return (
        f"Đã xác thực catalog và lưu đơn {saved['order_id']} cho {customer['name']}, "
        f"liên hệ {customer['phone']} / {customer['email']}, giao đến {customer['shipping_address']}. "
        f"Sản phẩm: {item_summary}. "
        f"Khuyến mãi hệ thống {discount['campaign_code']} ({rate_percent}%), "
        f"tổng thanh toán {final_total} VND. "
        f"File lưu tại {saved['save_path']}."
    )


def _normalize_for_rules(text: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    compact = re.sub(r"[^a-zA-Z0-9]+", " ", stripped.lower())
    return re.sub(r"\s+", " ", compact).strip()
