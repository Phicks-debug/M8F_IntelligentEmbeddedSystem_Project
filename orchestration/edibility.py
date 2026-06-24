SAFETY_WARNING = (
    "Do not consume wild mushrooms based only on this model prediction. "
    "Ask a qualified mycologist or local expert before eating any wild mushroom."
)

EDIBILITY = {
    "Agaricus_bisporus": {
        "label": "commonly edible",
        "note": "Cultivated button/cremini/portobello mushroom.",
    },
    "Amanita_muscaria": {
        "label": "toxic / not edible",
        "note": "Contains psychoactive/toxic compounds; do not eat.",
    },
    "Boletus_edulis": {
        "label": "commonly edible",
        "note": "Porcini/cep; edible when correctly identified.",
    },
    "Cantharellus_cibarius": {
        "label": "commonly edible",
        "note": "Chanterelle; has poisonous lookalikes, so identification matters.",
    },
    "Coprinus_comatus": {
        "label": "edible when young",
        "note": "Shaggy mane; generally eaten young before the gills blacken.",
    },
    "Laetiporus_sulphureus": {
        "label": "edible when young/cooked",
        "note": "Chicken-of-the-woods; can cause adverse reactions and should not be eaten raw.",
    },
    "Morchella_esculenta": {
        "label": "edible only when cooked",
        "note": "Morel; should not be eaten raw and must be prepared correctly.",
    },
    "Pleurotus_ostreatus": {
        "label": "commonly edible",
        "note": "Oyster mushroom; widely cultivated and eaten.",
    },
}


def edibility_for(class_name):
    return EDIBILITY.get(
        class_name,
        {
            "label": "unknown edibility",
            "note": "No edibility mapping is available for this class.",
        },
    )
