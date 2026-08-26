import pandas as pd, re, collections

SRC = "/Users/umut/Desktop/HARVARD/SUMMER/BCH_CHEST_CT_ARC/just_reports.xlsx"
raw = pd.read_excel(SRC)["Report Text"].fillna("").str.replace("_x000D_", "", regex=False)

def sect(text, name):
    m = re.search(rf"^{name}\s*:?\s*(.*?)(?=^[A-Z][A-Z /&\-]{{2,40}}:|\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else ""

NEG = re.compile(r"\b(no|not|nor|neither|without|negative for|free of|absence of|resolved|"
                 r"resolution of|rather than|unlikely|denies|rule out|evaluate for|assess for|"
                 r"screen for|surveillance for|history of|h/o|prior)\b", re.I)

def positive(text, rx):
    for m in rx.finditer(text):
        win = re.split(r"[.;:*\n]", text[max(0, m.start() - 60):m.start()])[-1]
        if not NEG.search(win):
            return True
    return False

# ---- CT-RATE's 18 labels, regexes written for pediatric report wording ----
CTRATE = [
    ("Medical material",                r"port-?a-?cath|portacath|central (?:venous )?(?:line|catheter)|\bpicc\b|tracheostomy|chest tube|thoracostomy|pacemaker|\bstent\b|surgical clip|sternal wire|\bcoil\b|g-?tube|feeding tube|endotracheal tube|\bicd\b|catheter|prosthe|\bmesh\b|drain\b"),
    ("Arterial wall calcification",     r"(?:aortic|arterial|aorta)[^.\n]{0,40}calcifi|calcifi[^.\n]{0,40}(?:aorta|aortic|artery|arterial)"),
    ("Cardiomegaly",                    r"cardiomegaly|enlarged (?:cardiac silhouette|heart)|heart size is (?:mildly |moderately |markedly )?(?:enlarged|increased)|cardiac (?:enlargement|silhouette is enlarged)|ventricular (?:enlargement|dilat)|mild cardiac enlargement"),
    ("Pericardial effusion",            r"pericardial (?:effusion|fluid)"),
    ("Coronary artery wall calcification", r"coronary[^.\n]{0,30}calcifi|calcifi[^.\n]{0,30}coronary"),
    ("Hiatal hernia",                   r"hiatal hernia|hiatus hernia"),
    ("Lymphadenopathy",                 r"lymphadenopathy|adenopathy|enlarged (?:mediastinal |hilar |axillary )?(?:lymph )?nodes?|prominent (?:lymph )?nodes?|borderline (?:enlarged )?(?:lymph )?node|pathologic(?:ally enlarged)? node"),
    ("Emphysema",                       r"emphysema|bulla\b|bullae|bullous|pneumatocele"),
    ("Atelectasis",                     r"atelecta|(?:lobar|segmental|subsegmental) collapse|collapsed (?:lobe|lung)"),
    ("Lung nodule",                     r"\bnodule|nodular (?:opacit|densit|focus|foci)|micronodul"),
    ("Lung opacity",                    r"opacit|ground[- ]?glass|\bGGO\b|infiltrat|airspace disease|density in the (?:right|left) (?:upper|middle|lower) lobe"),
    ("Pulmonary fibrotic sequela",      r"fibrosi|fibrotic|\bscar(?:ring)?\b|architectural distortion|traction bronchiect|honeycomb|reticulation|post[- ]?(?:inflammatory|infectious) (?:change|scar)"),
    ("Peribronchial thickening",        r"peribronchial thickening|bronchial wall thickening|airway wall thickening|bronchial thickening|peribronchovascular thickening"),
    ("Consolidation",                   r"consolidat"),
    ("Bronchiectasis",                  r"bronchiecta"),
    ("Interlobular septal thickening",  r"(?:interlobular )?septal thickening|interstitial thickening|smooth interlobular|crazy[- ]paving"),
    ("Mosaic attenuation pattern",      r"mosaic (?:attenuation|perfusion|pattern)|air trapping|heterogeneous (?:lung )?attenuation"),
    ("Pleural effusion",                r"pleural effusion|pleural fluid|hydrothorax|hemothorax|(?:small|moderate|large) effusion"),
]

# ---- pathologies present in pediatric corpus but NOT in CT-RATE's 18 ----
PEDS_ONLY = [
    ("Pulmonary metastases (explicit)", r"metasta"),
    ("Mass / neoplasm (thoracic)",      r"\bmass\b|neoplas|\btumor\b|sarcoma|lymphoma|blastoma|carcinoma|\bPNET\b"),
    ("Pneumothorax",                    r"pneumothora"),
    ("Tree-in-bud (small-airways infl.)", r"tree[- ]in[- ]bud"),
    ("Mucus plugging",                  r"mucous plug|mucus plug|mucoid impaction|mucus[- ]filled|plugging"),
    ("Pulmonary cyst / cystic change",  r"\bcyst"),
    ("Cavitation",                      r"cavitat|cavitary"),
    ("Calcified granuloma",             r"granulom|calcified (?:lung )?nodule"),
    ("Pleural thickening / pleural nodule", r"pleural thickening|pleural nodul|pleural (?:mass|deposit)"),
    ("Bone lesion / fracture",          r"fractur|lytic lesion|blastic lesion|bone lesion|osseous lesion|rib lesion|marrow (?:signal|replacement)"),
    ("Chest wall deformity (pectus/scoliosis)", r"pectus|scolios|kyphos|chest wall deformity|thoracic (?:cage )?deformity"),
    ("Post-surgical / post-treatment change", r"post-?surgical|post-?operative|postsurgical|post-?treatment|post-?radiation|resection|lobectomy|wedge (?:resection|excision)|thoracotomy|sternotomy|metastasectomy"),
    ("Thymic abnormality",              r"thymic (?:mass|enlarge|hyperplas|rebound)|thymoma|prominent thymus"),
    ("Congenital vascular anomaly",     r"aberrant (?:right |left )?subclavian|vascular ring|double aortic arch|right aortic arch|anomalous (?:pulmonary )?(?:vein|venous|artery)|persistent left superior vena cava|pulmonary sling|\bTAPVC\b|\bPAPVR\b"),
    ("Congenital lung malformation",    r"\bCPAM\b|sequestration|congenital lobar|bronchogenic cyst|pulmonary hypoplasia|lung agenesis|congenital diaphragmatic hernia|\bCDH\b"),
    ("Tracheo-bronchomalacia / airway narrowing", r"malacia|tracheomalacia|bronchomalacia|tracheal (?:narrowing|stenosis)|bronchial (?:narrowing|stenosis)|subglottic stenosis|airway narrowing"),
    ("Pulmonary embolism",              r"pulmonary embol|filling defect in the (?:pulmonary|right|left) (?:artery|arteries)"),
    ("Septic emboli",                   r"septic embol"),
    ("Fungal / invasive aspergillosis",  r"aspergill|fungal|halo sign|angioinvasive|mucormyc"),
    ("Diffuse lung disease of infancy (NEHI/ChILD)", r"\bNEHI\b|neuroendocrine cell hyperplasia|\bChILD\b|children'?s interstitial lung disease|surfactant (?:dysfunction|deficiency)|follicular bronchiolitis"),
    ("Bronchiolitis obliterans / post-transplant", r"bronchiolitis obliterans|\bBOS\b|\bGVHD\b|graft[- ]versus[- ]host|obliterative bronchiolitis|constrictive bronchiolitis"),
    ("Hernia (non-hiatal, e.g. diaphragmatic/Bochdalek/Morgagni)", r"(?:diaphragmatic|bochdalek|morgagni|lung) hernia|herniation"),
    ("Esophageal abnormality",          r"esophageal (?:dilat|thickening|mass|wall thickening)|dilated esophagus|esophagitis"),
    ("Pulmonary hypertension signs",    r"pulmonary (?:arterial )?hypertension|enlarged (?:main )?pulmonary artery|pulmonary artery (?:dilat|enlarge)"),
    ("Lymphatic malformation / chylothorax", r"lymphatic malformation|lymphangiomatosis|chylothorax|chylous|thoracic duct"),
    ("Sickle-cell / chronic marrow change", r"sickle cell|acute chest syndrome|marrow expansion|\bH-?shaped vertebra"),
]

# ---- run ----
imp_hits, body_hits = collections.Counter(), collections.Counter()
rows = []
for t in raw:
    imp = re.sub(r"END OF IMPRESSION.*", "", sect(t, "IMPRESSION"), flags=re.S | re.I)
    body = sect(t, "FINDINGS") + "\n" + imp
    rec = {}
    for lbl, pat in CTRATE + PEDS_ONLY:
        rx = re.compile(pat, re.I)
        b = positive(body, rx)
        i = positive(imp, rx)
        if b: body_hits[lbl] += 1
        if i: imp_hits[lbl] += 1
        rec[lbl] = int(b)
    rows.append(rec)

N = len(raw)
out = pd.DataFrame(rows)
out.to_csv("/private/tmp/claude-501/-Users-umut-Desktop-HARVARD-SUMMER-BCH-CHEST-CT-ARC/8eace5da-de45-4b0d-9824-9b7b8da0578e/scratchpad/ctrate_labels_peds.csv", index=False)

print(f"N = {N}\n")
print("### CT-RATE 18 LABELS -> PEDIATRIC PREVALENCE ###")
print(f"{'label':38s} {'body_n':>7s} {'body_%':>7s} {'imp_n':>6s} {'imp_%':>6s}")
for lbl, _ in CTRATE:
    print(f"{lbl:38s} {body_hits[lbl]:7d} {body_hits[lbl]/N*100:6.1f}% {imp_hits[lbl]:6d} {imp_hits[lbl]/N*100:5.1f}%")

print("\n### PEDIATRIC-ONLY (not in CT-RATE 18) ###")
print(f"{'label':60s} {'body_n':>7s} {'body_%':>7s} {'imp_n':>6s} {'imp_%':>6s}")
for lbl, _ in PEDS_ONLY:
    print(f"{lbl:60s} {body_hits[lbl]:7d} {body_hits[lbl]/N*100:6.1f}% {imp_hits[lbl]:6d} {imp_hits[lbl]/N*100:5.1f}%")

print("\n### CO-OCCURRENCE of top CT-RATE labels (body) ###")
top = [l for l, _ in CTRATE if body_hits[l] >= 60]
print(out[top].corr().round(2).to_string())
