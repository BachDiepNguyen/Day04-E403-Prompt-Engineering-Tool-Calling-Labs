from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProductRecord(BaseModel):
    product_id: str
    sku: str
    name: str
    category: str
    brand: str
    unit_price: int
    stock: int
    warranty_months: int
    tags: list[str] = Field(default_factory=list)
    description: str


class OrderLineInput(BaseModel):
    product_id: str = Field(..., description="Exact stable product ID from catalog/tool output, for example LT-001.")
    quantity: int = Field(..., ge=1, description="Requested quantity. Use 1 when the user lists or quotes an item without a number.")


class ListProductsInput(BaseModel):
    query: str | None = Field(default=None, description="Free-text catalog search using requested product names, brands, or feature hints.")
    category: str | None = Field(default=None, description="Optional category filter such as laptop, monitor, mouse, keyboard, dock, storage, stand, webcam, or headphone.")
    max_unit_price: int | None = Field(default=None, ge=0, description="Optional maximum unit price in VND.")
    required_tags: list[str] = Field(
        default_factory=list,
        description="Optional tags or feature hints such as wireless, office, gaming, or anc.",
    )
    in_stock_only: bool = Field(default=True, description="When true, return only products with stock > 0.")
    limit: int = Field(default=8, ge=1, le=20, description="Maximum number of products to return.")


class ProductDetailInput(BaseModel):
    product_ids: list[str] = Field(
        ...,
        min_length=1,
        description="One or more exact product IDs returned by list_products. Call once with all selected IDs for the order.",
    )


class DiscountInput(BaseModel):
    seed_hint: str = Field(
        ...,
        description="Stable seed used to simulate campaign discount. Use customer email; fallback to phone only if email is unavailable.",
    )
    customer_tier: str = Field(default="standard", description="Customer segment. Use standard unless the user clearly states VIP.")


class CalculateTotalsInput(BaseModel):
    items: list[OrderLineInput] = Field(
        ...,
        min_length=1,
        description="Normalized line items using exact product IDs and quantities.",
    )
    detail_token: str = Field(
        ...,
        description="Validation token returned by get_product_details for exactly this product set.",
    )
    discount_rate: float = Field(..., description="Discount rate returned by get_discount only. Supported values are 0.1 or 0.2.")


class SaveOrderInput(BaseModel):
    customer_name: str = Field(..., description="Customer full name.")
    customer_phone: str = Field(..., description="Customer phone number.")
    customer_email: str = Field(..., description="Customer email address.")
    shipping_address: str = Field(..., description="Shipping destination in free text.")
    items: list[OrderLineInput] = Field(..., min_length=1, description="Final validated product lines with exact IDs and quantities.")
    detail_token: str = Field(..., description="Validation token returned by get_product_details for exactly this product set.")
    discount_rate: float = Field(..., description="Discount rate returned by get_discount; never manually choose another value.")
    campaign_code: str = Field(..., description="Campaign code returned by get_discount.")
    customer_tier: str = Field(default="standard", description="Customer segment associated with the discount.")
    notes: str = Field(default="", description="Optional internal note. Keep it short.")


class ToolCallRecord(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    output: str = ""


class AgentResult(BaseModel):
    query: str
    final_answer: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    provider: str = "google"
    model_name: str | None = None
    saved_order: dict[str, Any] | None = None
    saved_order_path: str | None = None
