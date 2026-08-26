"""Label and region schemas, selected by RAC_SCHEMA.

Two schemas live here so the pediatric work cannot disturb the adult numbers.
`ctrate` is the frozen 18-label / 10-region set every released checkpoint was
trained on; `peds` is the 27-label / 10-region set for the pediatric adaptation.
Default is `ctrate`, so a checkout with no env set behaves exactly as before.

Why 27 and not 23. Dropping the four classes that are near-absent in children
(coronary and arterial calcification, hiatal hernia, emphysema) was the earlier
plan, and it is wrong once the same model has to keep serving adults: those are
among its strongest adult classes. A head that is deleted is not forgotten, it
is gone. They cost nine columns of near-constant zeros on the pediatric side,
which is cheap.

Why the pediatric regions are 10 and not 13. Measured over 8816 volumes at the
12x12x12 feature grid, 6 of the 10 adult regions fell below 98% presence in
children - vessels 82.6%, esophagus 83.3%, trachea 89.3%. An absent region does
not raise; anatomy_qformer turns it into an unrestricted global query, so the
query silently stops being an anatomy query. Merging was then measured rather
than assumed: heart+aorta+vessels reaches 99.1% because those three fail on
*different* volumes, while trachea+esophagus only reaches 97.2% because they
fail on the same small children. The five-way merge scores highest (99.8%) and
is still rejected: Peribronchial thickening and Bronchiectasis route to the
airway, and putting the trachea in the same region as the heart would let a
bronchial-wall query attend to myocardium.
"""
from __future__ import annotations
import os

CTRATE_PATHOLOGIES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]

PEDS_PATHOLOGIES = CTRATE_PATHOLOGIES + [
    "Post-surgical or post-treatment change",
    "Pulmonary metastases",
    "Tree-in-bud",
    "Pulmonary cyst",
    "Mass or neoplasm",
    "Mucus plugging",
    "Pleural thickening or nodule",
    "Bone lesion or fracture",
    "Pneumothorax",
]

CTRATE_FINE_LABEL_NAMES = {
    0: "background", 1: "lung_upper_lobe_left", 2: "lung_lower_lobe_left",
    3: "lung_upper_lobe_right", 4: "lung_middle_lobe_right",
    5: "lung_lower_lobe_right", 6: "trachea", 7: "heart", 8: "aorta",
    9: "vessels", 10: "esophagus",
}

PEDS_FINE_LABEL_NAMES = {
    0: "background", 1: "lung_upper_lobe_left", 2: "lung_lower_lobe_left",
    3: "lung_upper_lobe_right", 4: "lung_middle_lobe_right",
    5: "lung_lower_lobe_right",
    6: "central_airway_and_esophagus",
    7: "heart_and_great_vessels",
    8: "pleural_space",
    9: "chest_wall_and_skeleton",
    10: "upper_abdomen",
}

CTRATE_PATHOLOGY_FINE_ORGANS = {
    "Medical material": [], "Arterial wall calcification": [8], "Cardiomegaly": [7],
    "Pericardial effusion": [7], "Coronary artery wall calcification": [7, 8],
    "Hiatal hernia": [10], "Lymphadenopathy": [9], "Emphysema": [1, 3],
    "Atelectasis": [1, 2, 3, 4, 5], "Lung nodule": [1, 2, 3, 4, 5],
    "Lung opacity": [1, 2, 3, 4, 5], "Pulmonary fibrotic sequela": [2, 5],
    "Pleural effusion": [2, 5], "Mosaic attenuation pattern": [1, 2, 3, 4, 5],
    "Peribronchial thickening": [1, 2, 3, 4, 5, 6], "Consolidation": [1, 2, 3, 4, 5],
    "Bronchiectasis": [1, 2, 3, 4, 5], "Interlobular septal thickening": [1, 2, 3, 4, 5],
}

# Region ids 6-10 changed meaning, so every routing that used them had to be
# re-derived. Under the adult table, "Arterial wall calcification": [8] would
# now hard-mask that query to the pleural shell and "Hiatal hernia": [10] to the
# upper abdomen - valid ids, no error, anatomically wrong supervision.
# The rule for anything with no segmentable home is []: unrestricted attention
# beats a confident mask that excludes the target.
PEDS_PATHOLOGY_FINE_ORGANS = {
    "Medical material": [],                       # devices span airway/vessel/wall
    "Arterial wall calcification": [7],           # aorta now lives in 7
    "Cardiomegaly": [7],
    "Pericardial effusion": [7],
    "Coronary artery wall calcification": [7],
    "Hiatal hernia": [6, 10],                     # esophagus is in 6; hernia reaches 10
    "Lymphadenopathy": [],                        # mediastinal/hilar/axillary, distributed
    "Emphysema": [1, 2, 3, 4, 5],
    "Atelectasis": [1, 2, 3, 4, 5],
    "Lung nodule": [1, 2, 3, 4, 5],
    "Lung opacity": [1, 2, 3, 4, 5],
    "Pulmonary fibrotic sequela": [1, 2, 3, 4, 5],
    "Pleural effusion": [8, 2, 5],                # the pleural shell, plus dependent lung
    "Mosaic attenuation pattern": [1, 2, 3, 4, 5],
    "Peribronchial thickening": [1, 2, 3, 4, 5, 6],
    "Consolidation": [1, 2, 3, 4, 5],
    "Bronchiectasis": [1, 2, 3, 4, 5, 6],
    "Interlobular septal thickening": [1, 2, 3, 4, 5],
    "Post-surgical or post-treatment change": [],  # lung+pleura+wall+mediastinum at once
    "Pulmonary metastases": [1, 2, 3, 4, 5],
    "Tree-in-bud": [1, 2, 3, 4, 5],
    "Pulmonary cyst": [1, 2, 3, 4, 5],
    "Mass or neoplasm": [],                        # thymic/anterior mediastinal: no mask
    "Mucus plugging": [1, 2, 3, 4, 5, 6],
    "Pleural thickening or nodule": [8],
    "Bone lesion or fracture": [9],
    "Pneumothorax": [8],
}

PEDS_REGION_KEYWORDS = {
    1: ["left upper lobe", "lingula", "left apex"],
    2: ["left lower lobe", "left base"],
    3: ["right upper lobe", "right apex"],
    4: ["right middle lobe"],
    5: ["right lower lobe", "right base"],
    6: ["trachea", "tracheal", "bronchus", "bronchi", "carina", "airway",
        "esophagus", "esophageal"],
    7: ["heart", "cardiac", "pericardi", "aorta", "aortic", "vena cava",
        "pulmonary artery", "pulmonary vein", "great vessel", "mediastinal vessel"],
    8: ["pleura", "pleural", "effusion", "pneumothorax", "costophrenic", "fissur"],
    9: ["rib", "ribs", "sternum", "sternal", "vertebra", "spine", "spinal",
        "scoliosis", "clavicle", "scapula", "chest wall", "soft tissue",
        "subcutaneous", "osseous", "bone", "bony", "fracture", "axilla", "muscle"],
    10: ["liver", "hepatic", "spleen", "splenic", "kidney", "renal", "adrenal",
         "upper abdomen", "abdominal", "stomach", "gastric", "pancrea", "gallbladder"],
}


def active() -> dict:
    """Resolve the schema named by RAC_SCHEMA (default: ctrate)."""
    name = os.environ.get("RAC_SCHEMA", "ctrate").lower()
    if name == "peds":
        return {
            "name": "peds",
            "PATHOLOGIES": list(PEDS_PATHOLOGIES),
            "FINE_LABEL_NAMES": dict(PEDS_FINE_LABEL_NAMES),
            "PATHOLOGY_FINE_ORGANS": {k: list(v) for k, v in PEDS_PATHOLOGY_FINE_ORGANS.items()},
            "REGION_KEYWORDS": {k: list(v) for k, v in PEDS_REGION_KEYWORDS.items()},
        }
    if name != "ctrate":
        raise ValueError("RAC_SCHEMA must be 'ctrate' or 'peds', got %r" % name)
    return {
        "name": "ctrate",
        "PATHOLOGIES": list(CTRATE_PATHOLOGIES),
        "FINE_LABEL_NAMES": dict(CTRATE_FINE_LABEL_NAMES),
        "PATHOLOGY_FINE_ORGANS": {k: list(v) for k, v in CTRATE_PATHOLOGY_FINE_ORGANS.items()},
        "REGION_KEYWORDS": None,        # dataset.py keeps its own for the adult path
    }
