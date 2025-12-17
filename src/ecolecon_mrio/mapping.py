"""Mappings for sector codes to human-readable labels."""

# OECD ICIO 2023 sector labels (short)
SECTOR_LABELS_ICIO = {
    "A01_02": "Agriculture",
    "A03": "Forestry & fishing",
    "B05_06": "Coal & lignite",
    "B07_08": "Metal ores",
    "B09": "Other mining",
    "C10T12": "Food & beverages",
    "C13T15": "Textiles & apparel",
    "C16": "Wood products",
    "C17_18": "Paper & printing",
    "C19": "Petroleum & coke",
    "C20": "Chemicals",
    "C21": "Pharmaceuticals",
    "C22": "Rubber & plastics",
    "C23": "Non-metallic minerals",
    "C24": "Basic metals",
    "C25": "Fabricated metals",
    "C26": "Electronics",
    "C27": "Electrical equipment",
    "C28": "Machinery",
    "C29": "Motor vehicles",
    "C30": "Other transport equip",
    "C31T33": "Other manufacturing",
    "D": "Electricity & gas",
    "E": "Water & waste",
    "F": "Construction",
    "G": "Wholesale & retail",
    "H49": "Land transport",
    "H50": "Water transport",
    "H51": "Air transport",
    "H52": "Warehousing",
    "H53": "Postal",
    "I": "Accommodation & food",
    "J58T60": "Publishing & AV",
    "J61": "Telecom",
    "J62_63": "IT services",
    "K": "Finance & insurance",
    "L": "Real estate",
    "M": "Professional services",
    "N": "Administrative services",
    "O": "Public administration",
    "P": "Education",
    "Q": "Health & social",
    "R": "Arts & recreation",
    "S": "Other services",
    "T": "Households",
}


def sector_label(code: str, include_code: bool = True, region: str | None = None) -> str:
    """Return a human-readable label for a sector code, optionally with region."""
    name = SECTOR_LABELS_ICIO.get(code, code)
    base = f"{code} {name}" if include_code and code != name else name
    if region:
        return f"{region}-{base}"
    return base
