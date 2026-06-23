"""
Species-to-edibility labels for the classification model.

These labels are for project reporting and demo output only. Mushroom
edibility is safety-critical and depends on correct identification,
freshness, preparation, geography, lookalikes, and individual reactions.
Do not use this project as a consumption guide.
"""

from __future__ import annotations

from dataclasses import dataclass


SAFETY_WARNING = (
    "Do not consume wild mushrooms based only on this model prediction. "
    "Ask a qualified mycologist or local expert before eating any wild mushroom."
)


@dataclass(frozen=True)
class EdibilityInfo:
    """Human-readable edibility metadata for one predicted species."""

    category: str
    label: str
    note: str


EDIBILITY_BY_CLASS: dict[str, EdibilityInfo] = {
    "Agaricus_bisporus": EdibilityInfo(
        category="edible",
        label="commonly edible",
        note="Cultivated button/cremini/portobello mushroom.",
    ),
    "Amanita_muscaria": EdibilityInfo(
        category="toxic",
        label="toxic / not edible",
        note="Contains psychoactive/toxic compounds; do not eat.",
    ),
    "Boletus_edulis": EdibilityInfo(
        category="edible",
        label="commonly edible",
        note="Porcini/cep; edible when correctly identified.",
    ),
    "Cantharellus_cibarius": EdibilityInfo(
        category="edible",
        label="commonly edible",
        note="Chanterelle; has poisonous lookalikes, so identification matters.",
    ),
    "Coprinus_comatus": EdibilityInfo(
        category="edible_with_caution",
        label="edible when young",
        note="Shaggy mane; generally eaten young before the gills blacken.",
    ),
    "Laetiporus_sulphureus": EdibilityInfo(
        category="edible_with_caution",
        label="edible when young/cooked",
        note="Chicken-of-the-woods; can cause adverse reactions and should not be eaten raw.",
    ),
    "Morchella_esculenta": EdibilityInfo(
        category="edible_with_caution",
        label="edible only when cooked",
        note="Morel; should not be eaten raw and must be prepared correctly.",
    ),
    "Pleurotus_ostreatus": EdibilityInfo(
        category="edible",
        label="commonly edible",
        note="Oyster mushroom; widely cultivated and eaten.",
    ),
}


def get_edibility(class_name: str) -> EdibilityInfo:
    """Return edibility metadata for a model class name."""

    return EDIBILITY_BY_CLASS.get(
        class_name,
        EdibilityInfo(
            category="unknown",
            label="unknown edibility",
            note="No edibility mapping is available for this class.",
        ),
    )


def validate_edibility_map(class_names: list[str]) -> None:
    """Raise if any model classes are missing from the edibility mapping."""

    missing = sorted(set(class_names) - set(EDIBILITY_BY_CLASS))
    if missing:
        raise ValueError(f"Missing edibility mapping for classes: {missing}")
