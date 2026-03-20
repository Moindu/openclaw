"""Shopware 6 Store-API product importer."""
import os
import re

import httpx

from config import SHOPWARE_BASE_URL, COLLECTION_PRODUCTS
from embeddings import get_embeddings_batch
from indexer import get_chroma_client, get_or_create_collection


def fetch_products(
    access_key: str | None = None,
    base_url: str = SHOPWARE_BASE_URL,
    limit: int = 100,
) -> list[dict]:
    """Fetch all products from Shopware Store-API."""
    key = access_key or os.environ.get("SHOPWARE_ACCESS_KEY", "")
    if not key:
        raise ValueError("SHOPWARE_ACCESS_KEY not set")

    headers = {
        "sw-access-key": key,
        "Content-Type": "application/json",
    }

    all_products = []
    page = 1

    while True:
        response = httpx.post(
            f"{base_url}/product",
            headers=headers,
            json={
                "limit": limit,
                "page": page,
                "associations": {
                    "categories": {},
                    "properties": {"associations": {"group": {}}},
                },
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        elements = data.get("elements", [])
        all_products.extend(elements)

        if len(elements) < limit:
            break
        page += 1

    return all_products


def product_to_text(product: dict) -> str:
    """Convert a Shopware product to searchable text."""
    parts = []

    # Shopware returns name in translated for variants
    name = product.get("name") or product.get("translated", {}).get("name", "")
    parts.append(f"Produkt: {name}")

    number = product.get("productNumber", "")
    if number:
        parts.append(f"Artikelnummer: {number}")

    description = product.get("description") or product.get("translated", {}).get("description") or ""
    if description:
        clean = re.sub(r"<[^>]+>", " ", description).strip()
        parts.append(f"Beschreibung: {clean}")

    price_info = product.get("calculatedPrice", {})
    price = price_info.get("totalPrice")
    if price is not None:
        parts.append(f"Preis: {price:.2f} EUR")

    stock = product.get("stock")
    if stock is not None:
        parts.append(f"Bestand: {stock}")

    categories = product.get("categories") or []
    cat_names = [c.get("name", "") for c in categories if c.get("name")]
    if cat_names:
        parts.append(f"Kategorien: {', '.join(cat_names)}")

    properties = product.get("properties") or []
    for prop in properties:
        group_name = prop.get("group", {}).get("name", "")
        prop_name = prop.get("name", "")
        if group_name and prop_name:
            parts.append(f"{group_name}: {prop_name}")

    return "\n".join(parts)


def index_products(access_key: str | None = None):
    """Fetch products from Shopware and index them in ChromaDB."""
    products = fetch_products(access_key)
    if not products:
        print("No products found.")
        return

    client = get_chroma_client()
    collection = get_or_create_collection(client, COLLECTION_PRODUCTS)

    texts = [product_to_text(p) for p in products]
    embeddings = get_embeddings_batch(texts)

    ids = [p.get("id", f"product_{i}") for i, p in enumerate(products)]
    metadatas = [
        {
            "name": p.get("name", ""),
            "productNumber": p.get("productNumber", ""),
            "price": str(p.get("calculatedPrice", {}).get("totalPrice", "")),
            "source": "shopware",
        }
        for p in products
    ]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Indexed {len(products)} products from Shopware.")


if __name__ == "__main__":
    index_products()
