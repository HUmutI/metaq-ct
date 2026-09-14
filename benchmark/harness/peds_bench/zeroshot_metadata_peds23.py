"""
Zero-shot prompt metadata for the 23-class BCH pediatric chest-CT benchmark.

Style follows HLIP/src/hlip/zeroshot_metadata_ct_rate.py exactly:
  CLASSNAMES  -- ordered exactly as the class columns of peds23_labels_valid.csv
  ORGANS      -- class -> template key (identical to CT-RATE for the 12 shared classes)
  TEMPLATES   -- organ key -> (lambda c: sentence,)
  PROMPTS     -- class -> (negative prompt, positive prompt), phrased for the ORGAN templates

VOLUME_PROMPTS reproduces, for the pediatric classes, what zeroshot_ct_rate.py does in
main() for the 'volume' template: when the template no longer names the organ, the organ
word is put back into the prompt itself (e.g. "Nodule" -> "Lung nodule").
"""

CLASSNAMES = [
    "Medical material",
    "Cardiomegaly",
    "Pericardial effusion",
    "Lymphadenopathy",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
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

ORGANS = {
    "Medical material": "chest",
    "Cardiomegaly": "heart",
    "Pericardial effusion": "heart",
    "Lymphadenopathy": "mediastinum",
    "Atelectasis": "lung",
    "Lung nodule": "lung",
    "Lung opacity": "lung",
    "Pulmonary fibrotic sequela": "lung",
    "Pleural effusion": "lung",
    "Mosaic attenuation pattern": "lung",
    "Peribronchial thickening": "lung",
    "Consolidation": "lung",
    "Bronchiectasis": "lung",
    "Interlobular septal thickening": "lung",
    "Post-surgical or post-treatment change": "chest",
    "Pulmonary metastases": "lung",
    "Tree-in-bud": "lung",
    "Pulmonary cyst": "lung",
    "Mass or neoplasm": "chest",
    "Mucus plugging": "lung",
    "Pleural thickening or nodule": "pleura",
    "Bone lesion or fracture": "bone",
    "Pneumothorax": "pleura",
}

TEMPLATES = {
    "lung": (lambda c: f'The lung shows: {c}.',),
    "heart": (lambda c: f'The heart shows: {c}.',),
    "mediastinum": (lambda c: f'The mediastinum shows: {c}.',),
    "pleura": (lambda c: f'The pleura shows: {c}.',),
    "bone": (lambda c: f'The bone shows: {c}.',),
    "chest": (lambda c: f'The chest shows: {c}.',),
    "volume": (lambda c: f'The volume shows: {c}.',),
}

PROMPTS = {
    "Medical material": ("Not medical material", "Medical material"),
    "Cardiomegaly": ("Not cardiomegaly", "Cardiomegaly"),
    "Pericardial effusion": ("Not pericardial effusion", "Pericardial effusion"),
    "Lymphadenopathy": ("Not lymphadenopathy", "Lymphadenopathy"),
    "Atelectasis": ("Not atelectatic", "Atelectasis"),
    "Lung nodule": ("Not nodule", "Nodule"),
    "Lung opacity": ("Not opacity", "Opacity"),
    "Pulmonary fibrotic sequela": ("Not pulmonary fibrotic sequela", "Pulmonary fibrotic sequela"),
    "Pleural effusion": ("Not pleural effusion", "Pleural effusion"),
    "Mosaic attenuation pattern": ("Not mosaic attenuation pattern", "Mosaic attenuation pattern"),
    "Peribronchial thickening": ("Not peribronchial thickening", "Peribronchial thickening"),
    "Consolidation": ("Not consolidation", "Consolidation"),
    "Bronchiectasis": ("Not bronchiectasis", "Bronchiectasis"),
    "Interlobular septal thickening": ("Not interlobular septal thickening", "Interlobular septal thickening"),
    "Post-surgical or post-treatment change": ("Not post-surgical or post-treatment change", "Post-surgical or post-treatment change"),
    "Pulmonary metastases": ("Not metastases", "Metastases"),
    "Tree-in-bud": ("Not tree-in-bud", "Tree-in-bud"),
    "Pulmonary cyst": ("Not cyst", "Cyst"),
    "Mass or neoplasm": ("Not mass or neoplasm", "Mass or neoplasm"),
    "Mucus plugging": ("Not mucus plugging", "Mucus plugging"),
    "Pleural thickening or nodule": ("Not thickening or nodule", "Thickening or nodule"),
    "Bone lesion or fracture": ("Not lesion or fracture", "Lesion or fracture"),
    "Pneumothorax": ("Not pneumothorax", "Pneumothorax"),
}

# applied when the template does not name the organ (i.e. --zeroshot-template != 'organ'),
# mirroring zeroshot_ct_rate.py::main()
VOLUME_PROMPTS = {
    "Lung nodule": ("Not lung nodule", "Lung nodule"),
    "Lung opacity": ("Not lung opacity", "Lung opacity"),
    "Pulmonary metastases": ("Not pulmonary metastases", "Pulmonary metastases"),
    "Pulmonary cyst": ("Not pulmonary cyst", "Pulmonary cyst"),
    "Pleural thickening or nodule": ("Not pleural thickening or nodule", "Pleural thickening or nodule"),
    "Bone lesion or fracture": ("Not bone lesion or fracture", "Bone lesion or fracture"),
}
