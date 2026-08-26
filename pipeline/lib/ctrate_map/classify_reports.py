import pandas as pd, re, collections, json, sys

SRC = "/Users/umut/Desktop/HARVARD/SUMMER/BCH_CHEST_CT_ARC/just_reports.xlsx"
df = pd.read_excel(SRC)
raw = df["Report Text"].fillna("").str.replace("_x000D_", "", regex=False)

def sect(text, name, nxt=None):
    m = re.search(rf"^{name}\s*:?\s*(.*?)(?=^[A-Z][A-Z /&\-]{{2,40}}:|\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else ""

# ---------- AGE ----------
AGE_PATS = [
    (r"\b(\d{1,2})\s*[- ]?\s*y\s*[./]?\s*o\b\.?", "y"),                      # 7yo, 7 y o, 7 y.o., 13y/o
    (r"\b(\d{1,2})\s*[- ]?\s*(?:yr|yrs|year|years)\s*[- ]?old\b", "y"),       # 7 year old, 7-year-old
    (r"\b(\d{1,2})\s*[- ]?\s*(?:yo|yof|yom)\b", "y"),                         # 7yof
    (r"\b(\d{1,2})\s*(?:F|M)\b(?=\s+(?:with|w/|s/p|presenting|patient|female|male|,))", "y"),  # 8F with ...
    (r"\b(\d{1,3})\s*[- ]?\s*(?:mo|month|months)\s*[- ]?old\b", "m"),
    (r"\b(\d{1,3})\s*[- ]?\s*m\s*[./]?\s*o\b\.?", "m"),
    (r"\b(\d{1,3})\s*[- ]?\s*(?:week|weeks|wk|wks)\s*[- ]?old\b", "w"),
    (r"\b(\d{1,3})\s*[- ]?\s*(?:day|days)\s*[- ]?old\b", "d"),
]

def get_age(ind, full):
    for scope in (ind, full):
        for pat, unit in AGE_PATS:
            m = re.search(pat, scope, re.I)
            if m:
                v = int(m.group(1))
                yrs = {"y": v, "m": v / 12.0, "w": v / 52.0, "d": v / 365.0}[unit]
                if yrs <= 100:
                    return yrs
        if re.search(r"\b(newborn|neonate|premature infant)\b", scope, re.I):
            return 0.05
    return None

def age_bin(a):
    if a is None: return "unknown"
    if a < 1: return "00 <1y (infant)"
    if a < 3: return "01 1-2y (toddler)"
    if a < 6: return "02 3-5y (preschool)"
    if a < 12: return "03 6-11y (school age)"
    if a < 18: return "04 12-17y (adolescent)"
    if a < 26: return "05 18-25y (young adult)"
    return "06 26+y (adult)"

# ---------- NEGATION ----------
NEG = re.compile(
    r"\b(no|not|without|negative for|free of|absence of|resolved|resolution of|"
    r"rather than|versus|unlikely|denies|r/o|rule out|evaluate for|assess for|"
    r"question of|screen for|surveillance for|concern for|history of|h/o|s/p|status post|prior)\b",
    re.I)
STOP = re.compile(r"[.;:*\n]|\b(but|however|with|there is|there are|demonstrat|shows|seen|noted|present)\b", re.I)

def is_negated(text, start):
    win = text[max(0, start - 60):start]
    # cut window at last sentence/clause break so negation doesn't leak across
    parts = re.split(r"[.;:*\n]", win)
    win = parts[-1]
    return bool(NEG.search(win))

FIND_KEYS = [
    ("Pulmonary nodule(s)",        r"nodul"),
    ("Ground-glass opacity",       r"ground[- ]?glass|GGO"),
    ("Consolidation",              r"consolidat"),
    ("Atelectasis",                r"atelecta"),
    ("Pleural effusion",           r"pleural effusion|effusion"),
    ("Pneumothorax",               r"pneumothora"),
    ("Lymphadenopathy",            r"lymphadenopathy|adenopathy|enlarged (?:lymph )?node|prominent lymph node"),
    ("Mass / neoplasm",            r"\bmass\b|neoplas|tumor|carcinoma|sarcoma|lymphoma|blastoma"),
    ("Metastatic disease",         r"metasta"),
    ("Scarring / fibrosis",        r"\bscar|fibrosi|fibrotic"),
    ("Bronchiectasis",             r"bronchiecta"),
    ("Bronchial wall thickening",  r"bronchial wall thickening|peribronchial thickening|airway wall thickening"),
    ("Air trapping / mosaic",      r"air trapping|mosaic attenuation|mosaic perfusion"),
    ("Cyst / cystic change",       r"\bcyst"),
    ("Septal / interstitial thickening", r"septal thickening|interstitial thickening|interlobular"),
    ("Tree-in-bud",                r"tree[- ]in[- ]bud"),
    ("Emphysema / bullae",         r"emphysema|bulla|bullous|pneumatocele"),
    ("Cavitation",                 r"cavitat|cavitary"),
    ("Calcified granuloma",        r"granulom|calcified nodule"),
    ("Pneumonia / infection",      r"pneumoni|infectio|abscess|empyema|septic emboli"),
    ("Cardiomegaly",               r"cardiomegaly|enlarged (?:cardiac|heart)|heart size is (?:mildly )?(?:enlarged|increased)"),
    ("Pericardial effusion",       r"pericardial effusion|pericardial fluid"),
    ("Vascular anomaly",           r"aberrant|vascular ring|anomalous|persistent left superior vena cava|pulmonary artery (?:dilat|enlarge)|aneurysm"),
    ("Pulmonary embolism",         r"pulmonary embol|filling defect"),
    ("Bone lesion / fracture",     r"fractur|lytic lesion|blastic lesion|bone lesion|osseous lesion"),
    ("Scoliosis / chest wall deform", r"scolios|pectus|kyphos|chest wall deformity"),
    ("Hernia",                     r"hernia"),
    ("Thymic abnormality",         r"thymic (?:mass|enlarge|hyperplas)|thymoma"),
    ("Esophageal abnormality",     r"esophageal (?:dilat|thickening|mass)|hiatal"),
    ("Pleural thickening",         r"pleural thickening|pleural nodul"),
    ("Mucus plugging",             r"mucous plug|mucus plug|mucoid impaction"),
    ("Lines / support devices",    r"port-a-cath|portacath|central (?:venous )?(?:line|catheter)|picc|tracheostomy|chest tube|g-tube|pacemaker"),
]
FIND_RE = [(lbl, re.compile(pat, re.I)) for lbl, pat in FIND_KEYS]

INDICATION_KEYS = [
    ("Oncology - solid tumor / sarcoma", r"rhabdomyosarcoma|osteosarcoma|ewing|wilms|neuroblastoma|hepatoblastoma|sarcoma|germ cell|teratoma|nephroblastoma|retinoblastoma|carcinoma|melanoma|solid tumor"),
    ("Oncology - leukemia / lymphoma",   r"leukemia|\ball\b|\baml\b|lymphoma|hodgkin|lymphoprolif|ptld"),
    ("Oncology - other / staging / surveillance", r"oncolog|malignan|cancer|metasta|staging|tumor|surveillance|restaging"),
    ("Pulmonary nodule follow-up",       r"nodule"),
    ("Infection / febrile neutropenia",  r"infect|fever|febrile|neutropen|pneumoni|aspergill|fungal|tubercul|abscess|sepsis"),
    ("Cystic fibrosis / bronchiectasis", r"cystic fibrosis|\bcf\b|bronchiectas"),
    ("Interstitial lung disease",        r"interstitial|\bild\b|child\b|fibrosi|hypersensitivity pneumonitis|sarcoid"),
    ("Transplant / GVHD",                r"transplant|\bgvhd\b|bmt|\bhsct\b|bronchiolitis obliterans"),
    ("Immunodeficiency",                 r"immunodefic|immunocompromis|\bscid\b|\bcvid\b"),
    ("Trauma",                           r"trauma|injur|\bmvc\b|fall\b|fracture"),
    ("Congenital / vascular anomaly",    r"congenital|vascular ring|aberrant|\bcpam\b|sequestration|malformation|anomal"),
    ("Chest wall / skeletal",            r"pectus|scolios|chest wall"),
    ("Airway / stridor / asthma",        r"stridor|asthma|airway|tracheo|bronchomalacia"),
    ("Hemoptysis / respiratory sx",      r"hemoptysis|cough|dyspnea|respiratory distress|shortness of breath|chest pain|wheez"),
]
IND_RE = [(lbl, re.compile(p, re.I)) for lbl, p in INDICATION_KEYS]

NORMAL_IMP = re.compile(
    r"normal chest ct|normal ct (?:of the )?chest|no acute (?:cardiopulmonary|intrathoracic|abnormalit)|"
    r"unremarkable (?:chest )?ct|normal (?:appearing )?(?:chest|examination|study)|"
    r"no evidence of (?:pulmonary )?(?:metasta|nodule)|no pulmonary nodul|clear lungs|lungs are clear|"
    r"no (?:acute |focal )?(?:abnormalit|findings)|negative (?:chest )?ct|within normal limits",
    re.I)
ABNORMAL_HINT = re.compile(
    r"nodul|opacit|consolidat|effusion|mass|metasta|adenopathy|lymphadenopath|bronchiecta|"
    r"atelecta|pneumothora|pneumoni|air trapping|ground[- ]?glass|fibrosi|cavit|scar|"
    r"abnormal|thickening|cyst|emphysem|infiltrat|lesion|collapse|hernia|enlarge", re.I)

rows = []
for i, t in enumerate(raw):
    ind = sect(t, "INDICATION")
    imp = sect(t, "IMPRESSION")
    imp = re.sub(r"END OF IMPRESSION.*", "", imp, flags=re.S | re.I)
    fnd = sect(t, "FINDINGS")
    body = fnd + "\n" + imp
    age = get_age(ind, t)

    pos = []
    for lbl, rx in FIND_RE:
        hit = False
        for m in rx.finditer(body):
            if not is_negated(body, m.start()):
                hit = True; break
        if hit: pos.append(lbl)

    # impression-only positives (higher confidence, drives healthy/unhealthy)
    imp_pos = []
    for lbl, rx in FIND_RE:
        for m in rx.finditer(imp):
            if not is_negated(imp, m.start()):
                imp_pos.append(lbl); break

    real = [p for p in imp_pos if p != "Lines / support devices"]
    imp_clean = imp.strip()
    normal_stmt = bool(NORMAL_IMP.search(imp_clean))
    stripped = re.sub(r"[^.]*?\b(no|not|nor|neither|never|without|negative for|free of|"
                      r"denies|unable to|resolution of|resolved)\b[^.]*\.", "", imp_clean, flags=re.I)
    abn_stmt = bool(ABNORMAL_HINT.search(stripped))
    explicit_normal = bool(re.match(r"\s*(?:\d[\.\)]\s*)?(?:normal (?:ct|chest|examination|study)|"
                                    r"normal ct of the chest|unremarkable)", imp_clean, re.I))

    if explicit_normal and not real:
        status = "Normal"
    elif real:
        status = "Abnormal"
    elif normal_stmt and not abn_stmt:
        status = "Normal"
    elif not abn_stmt:
        status = "Normal"
    else:
        status = "Abnormal"

    inds = [lbl for lbl, rx in IND_RE if rx.search(ind)]
    rows.append(dict(idx=i, age=age, age_bin=age_bin(age), status=status,
                     n_findings=len(real), findings=pos, imp_findings=real,
                     indications=inds, imp=imp_clean[:400], ind=ind[:250]))

out = pd.DataFrame(rows)
out.to_json("/private/tmp/claude-501/-Users-umut-Desktop-HARVARD-SUMMER-BCH-CHEST-CT-ARC/8eace5da-de45-4b0d-9824-9b7b8da0578e/scratchpad/classified.json", orient="records")

N = len(out)
print(f"TOTAL REPORTS: {N}\n")
print("=== NORMAL vs ABNORMAL ===")
for k, v in out.status.value_counts().items(): print(f"{k:12s} {v:5d}  {v/N*100:5.1f}%")

print("\n=== AGE GROUPS ===")
ab = out.age_bin.value_counts().sort_index()
for k, v in ab.items(): print(f"{k:26s} {v:5d}  {v/N*100:5.1f}%")
known = out[out.age.notna()]
print(f"\nage known: {len(known)} ({len(known)/N*100:.1f}%)  median {known.age.median():.1f}y  mean {known.age.mean():.1f}y  range {known.age.min():.2f}-{known.age.max():.0f}y")

print("\n=== FINDINGS (impression-level, positive) ===")
c = collections.Counter(f for r in rows for f in r["imp_findings"])
for k, v in c.most_common(): print(f"{k:36s} {v:5d}  {v/N*100:5.1f}%")

print("\n=== FINDINGS (whole report body, positive) ===")
c2 = collections.Counter(f for r in rows for f in r["findings"])
for k, v in c2.most_common(): print(f"{k:36s} {v:5d}  {v/N*100:5.1f}%")

print("\n=== INDICATION CATEGORIES ===")
c3 = collections.Counter(f for r in rows for f in r["indications"])
for k, v in c3.most_common(): print(f"{k:44s} {v:5d}  {v/N*100:5.1f}%")
print(f"{'(no indication keyword matched)':44s} {sum(1 for r in rows if not r['indications']):5d}")

print("\n=== BURDEN: # distinct impression findings per report ===")
for k, v in out.n_findings.value_counts().sort_index().items(): print(f"{k} finding(s): {v:5d}  {v/N*100:5.1f}%")

print("\n=== ABNORMAL RATE BY AGE GROUP ===")
x = out.groupby("age_bin").agg(n=("status", "size"), abn=("status", lambda s: (s == "Abnormal").sum()))
x["pct"] = (x.abn / x.n * 100).round(1)
print(x.to_string())
