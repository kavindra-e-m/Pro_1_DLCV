"""Conservative handbag scope filter using ABO product type and item title."""
import re

EXCLUDED_TYPES = {
    "BACKPACK", "LUGGAGE", "SUITCASE", "WALLET", "CAMERA_BAG",
    "CAMERA_BAGS_AND_CASES", "BRIEFCASE", "DUFFEL_BAG", "TRAVEL_BAG",
}
FASHION_TERMS = re.compile(
    r"\b(handbags?|purses?|totes?|shoulder bags?|clutches?|crossbody|cross-body|"
    r"satchels?|evening bags?|sling bags?)\b", re.I,
)
EXCLUSION_TERMS = re.compile(
    r"\b(backpacks?|rucksacks?|suitcases?|luggage|camera bags?|camera cases?|"
    r"wallets?|shopping carts?|shopping (bags?|totes?)|grocery|produce bags?|laptop bags?)\b", re.I,
)

def normalize(value):
    if isinstance(value, list):
        return " ".join(str(x.get("value", "") if isinstance(x, dict) else x) for x in value)
    return str(value or "")

def is_target_handbag(product_type, item_name="", category_paths=""):
    """Use a positive target product type, or strong title evidence for generic BAG.

    Explicit unrelated ABO product types always take precedence over incidental title
    mentions. Generic TOTE_BAG and BAG require target fashion-product title evidence.
    """
    kind = normalize(product_type).upper().replace(" ", "_")
    title = normalize(item_name)
    paths = normalize(category_paths)
    if kind in EXCLUDED_TYPES:
        return False
    if kind == "HANDBAG":
        return True
    if kind in {"BAG", "TOTE_BAG", "PURSE", "CROSSBODY_BAG", "CLUTCH"}:
        return bool(FASHION_TERMS.search(title) and not EXCLUSION_TERMS.search(title))
    # Category path is actual source metadata, not a generated visual label.
    if not kind and re.search(r"handbag|purse|shoulder bag|clutch|crossbody|tote", paths, re.I):
        return not bool(EXCLUSION_TERMS.search(title))
    return False
