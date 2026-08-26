# CT-RATE (adult) labels vs BCH pediatric chest CT corpus

Purpose: decide which of the 18 CT-RATE pathology labels transfer to pediatric chest CT,
which are dead weight, and which pediatric pathologies are missing from the label set.

- Pediatric corpus: `just_reports.xlsx`, **1206** reports (BCH chest CT).
- Adult label set: CT-RATE 18 multi-label pathologies (the model you trained).
- Method: regex + negation filter over FINDINGS + IMPRESSION sections. Multi-label, so
  percentages do not sum to 100. Prevalence = share of the 1206 reports with a positive
  (non-negated) mention. Numbers are text-mining estimates, not radiologist labels.

---

## 1. Headline numbers

| Quantity | N | % |
|---|---|---|
| Pediatric reports total | 1206 | 100% |
| Abnormal (any positive finding) | 992 | 82.3% |
| Reports with ≥1 CT-RATE label | 934 | 77.4% |
| Abnormal reports with **zero** CT-RATE labels (blind spot) | 67 | 6.8% of abnormal |
| ...of those, ≥1 pediatric-only label present | 52 | — |
| Mean CT-RATE labels per report | 1.61 | — |

CT-RATE-18 labels per pediatric report: 0 → 272, 1 → 458, 2 → 206, 3 → 132, 4 → 68,
5 → 42, 6 → 14, 7 → 8, 8 → 4, 9 → 2.

**Read:** the adult label set already covers ~77% of pediatric studies. The gap is not
mostly "unknown disease" — it is (a) pediatric-specific entities and (b) one adult label,
Lung nodule, absorbing half the pediatric corpus.

---

## 2. The 18 CT-RATE labels in pediatric data

`body` = FINDINGS+IMPRESSION, `imp` = IMPRESSION only (higher confidence / clinically salient).

| # | CT-RATE label | body N | body % | imp N | imp % | Verdict |
|---|---|---|---|---|---|---|
| 1 | Lung nodule | 602 | 49.9% | 599 | 49.7% | **Keep — dominant** |
| 2 | Lung opacity | 296 | 24.5% | 286 | 23.7% | **Keep — dominant** |
| 3 | Atelectasis | 176 | 14.6% | 160 | 13.3% | **Keep** |
| 4 | Peribronchial thickening | 153 | 12.7% | 147 | 12.2% | **Keep — more important in peds** |
| 5 | Pulmonary fibrotic sequela | 132 | 10.9% | 121 | 10.0% | Keep, but redefine (see §5) |
| 6 | Mosaic attenuation pattern | 122 | 10.1% | 120 | 10.0% | **Keep — more important in peds** |
| 7 | Bronchiectasis | 99 | 8.2% | 97 | 8.0% | **Keep — more important in peds** |
| 8 | Consolidation | 78 | 6.5% | 77 | 6.4% | Keep |
| 9 | Lymphadenopathy | 72 | 6.0% | 28 | 2.3% | Keep (weak; peds nodes often "prominent but normal") |
| 10 | Medical material | 65 | 5.4% | 44 | 3.6% | Keep |
| 11 | Pleural effusion | 51 | 4.2% | 51 | 4.2% | Keep (much rarer than adult) |
| 12 | Interlobular septal thickening | 32 | 2.7% | 29 | 2.4% | Keep, low-support |
| 13 | Pericardial effusion | 21 | 1.7% | 20 | 1.7% | Borderline |
| 14 | Cardiomegaly | 13 | 1.1% | 12 | 1.0% | Borderline |
| 15 | Emphysema | 13 | 1.1% | 13 | 1.1% | **Semantically wrong in peds** (see §5) |
| 16 | Hiatal hernia | 8 | 0.7% | 8 | 0.7% | **Drop / adult-only** |
| 17 | Arterial wall calcification | 4 | 0.3% | 3 | 0.2% | **Drop / adult-only** |
| 18 | Coronary artery wall calcification | 1 | 0.1% | 1 | 0.1% | **Drop / adult-only** |

---

## 3. Three buckets

### 3a. Shared — transfer directly (12 labels)
Same visual pattern, same wording, enough pediatric support to fine-tune.

| Label | Peds N | Peds % | Note |
|---|---|---|---|
| Lung nodule | 602 | 49.9% | Peds nodules are far smaller (1–5 mm micronodules) than adult CT-RATE nodules. Biggest distribution shift despite same name. |
| Lung opacity | 296 | 24.5% | Includes ground-glass (191, 15.8%) which dominates in peds. |
| Atelectasis | 176 | 14.6% | Peds often subsegmental/dependent, sedation-related. |
| Peribronchial thickening | 153 | 12.7% | Higher than adult relevance — CF/asthma cohort. |
| Pulmonary fibrotic sequela | 132 | 10.9% | In peds mostly post-infectious scarring, not IPF-type. |
| Mosaic attenuation pattern | 122 | 10.1% | In peds ≈ air trapping / small-airways disease. |
| Bronchiectasis | 99 | 8.2% | Driven by CF cohort (142 CF indications). |
| Consolidation | 78 | 6.5% | |
| Lymphadenopathy | 72 | 6.0% | Only 28 reach impression — peds size thresholds differ. |
| Medical material | 65 | 5.4% | Ports, PICCs, tracheostomy, stents. |
| Pleural effusion | 51 | 4.2% | Adult prevalence much higher; expect calibration drift. |
| Interlobular septal thickening | 32 | 2.7% | Low support; keep but expect weak AP. |

### 3b. Adult-only — near-absent in pediatric data (candidates to drop)
| Label | Peds N | Peds % | Why absent |
|---|---|---|---|
| Coronary artery wall calcification | 1 | 0.1% | Atherosclerosis does not exist in children. |
| Arterial wall calcification | 4 | 0.3% | Same; the 4 hits are dystrophic/post-treatment calcium. |
| Hiatal hernia | 8 | 0.7% | Degenerative/obesity-linked in adults. |
| Emphysema | 13 | 1.1% | Smoking-related emphysema absent; hits are **bullae / pneumatocele / congenital lobar overinflation** — different entity under the same word. |
| Cardiomegaly | 13 | 1.1% | Peds reports rarely call it; also CHD morphology ≠ adult cardiomegaly. |
| Pericardial effusion | 21 | 1.7% | Present but rare; keep only if effusion detection matters clinically. |

Dropping the top 4 removes ~0.5% of positive signal — they contribute almost nothing to
pediatric loss but cost you 4 output heads and 4 sources of false positives.

### 3c. Pediatric-only — missing from CT-RATE (candidates to add)
Sorted by prevalence. Everything ≥2% is worth a head.

| Label | Peds N | Peds % | Add? |
|---|---|---|---|
| Post-surgical / post-treatment change | 153 | 12.7% | **Yes** — 1 in 8 studies; oncology-surgery cohort. |
| Pulmonary metastases (explicit) | 103 | 8.5% | **Yes** — this is the corpus' clinical question. |
| Tree-in-bud (small-airways infection) | 88 | 7.3% | **Yes** — CF/immunocompromised. |
| Pulmonary cyst / cystic change | 87 | 7.2% | **Yes** |
| Mass / neoplasm (thoracic) | 72 | 6.0% | **Yes** |
| Mucus plugging | 52 | 4.3% | **Yes** — CF hallmark. |
| Pleural thickening / pleural nodule | 46 | 3.8% | **Yes** |
| Bone lesion / fracture | 30 | 2.5% | Yes (sarcoma mets to rib/spine). |
| Pneumothorax | 24 | 2.0% | Yes — actionable, cheap. |
| Fungal / invasive aspergillosis | 20 | 1.7% | Maybe — high clinical value, low N. |
| Calcified granuloma | 20 | 1.7% | Maybe — mainly to stop nodule false positives. |
| Bronchiolitis obliterans / post-transplant | 19 | 1.6% | Maybe — overlaps mosaic attenuation. |
| Chest wall deformity (pectus / scoliosis) | 13 | 1.1% | Low priority |
| Cavitation | 13 | 1.1% | Low priority |
| Pulmonary hypertension signs | 12 | 1.0% | Low priority |
| Thymic abnormality | 9 | 0.7% | Peds-specific but rare here |
| Tracheo-/bronchomalacia, airway narrowing | 7 | 0.6% | Rare in this corpus |
| Esophageal abnormality | 6 | 0.5% | Low |
| Diffuse lung disease of infancy (NEHI/ChILD) | 5 | 0.4% | Peds-exclusive entity, too rare here |
| Hernia (diaphragmatic/Bochdalek/Morgagni) | 4 | 0.3% | Rare |
| Congenital lung malformation (CPAM, sequestration) | 3 | 0.2% | Peds-exclusive, too rare here |
| Septic emboli | 2 | 0.2% | Rare |
| Pulmonary embolism | 2 | 0.2% | Rare |
| Congenital vascular anomaly | 2 | 0.2% | Under-counted; often phrased free-text |
| Lymphatic malformation / chylothorax | 1 | 0.1% | Rare |
| Sickle-cell / acute chest syndrome | 0 | 0.0% | Absent |

---

## 4. Most common pathologies, pediatric ranking (combined view)

| Rank | Pathology | N | % | In CT-RATE 18? |
|---|---|---|---|---|
| 1 | Lung nodule | 602 | 49.9% | yes |
| 2 | Lung opacity (incl. ground-glass) | 296 | 24.5% | yes |
| 3 | Atelectasis | 176 | 14.6% | yes |
| 4 | Peribronchial thickening | 153 | 12.7% | yes |
| 4= | Post-surgical / post-treatment change | 153 | 12.7% | **no** |
| 6 | Pulmonary fibrotic sequela / scarring | 132 | 10.9% | yes |
| 7 | Mosaic attenuation / air trapping | 122 | 10.1% | yes |
| 8 | Pulmonary metastases | 103 | 8.5% | **no** |
| 9 | Bronchiectasis | 99 | 8.2% | yes |
| 10 | Tree-in-bud | 88 | 7.3% | **no** |
| 11 | Pulmonary cyst | 87 | 7.2% | **no** |
| 12 | Consolidation | 78 | 6.5% | yes |
| 13 | Lymphadenopathy | 72 | 6.0% | yes |
| 13= | Mass / neoplasm | 72 | 6.0% | **no** |
| 15 | Medical material | 65 | 5.4% | yes |
| 16 | Mucus plugging | 52 | 4.3% | **no** |
| 17 | Pleural effusion | 51 | 4.2% | yes |
| 18 | Pleural thickening / nodule | 46 | 3.8% | **no** |

Adult top-performers on your table (pleural effusion 0.97, cardiomegaly 0.93, arterial/
coronary calcification 0.94/0.93) are exactly the labels with the **lowest** pediatric
prevalence. Your two weakest adult labels (lung nodule 0.72, pulmonary fibrotic sequela
0.72) are the two most prevalent pediatric ones. Expect overall pediatric AUROC to fall
below your adult numbers even after fine-tuning.

---

## 5. Same word, different disease — redefine before transfer

| Label | Adult meaning | Pediatric meaning in this corpus |
|---|---|---|
| Emphysema | Smoking-related destruction | Bullae, pneumatoceles, congenital lobar overinflation |
| Pulmonary fibrotic sequela | Post-IPF/ILD reticulation, honeycombing | Post-infectious/post-radiation scarring, linear bands |
| Mosaic attenuation | Small-vessel/chronic PE or ILD | Air trapping from small-airways disease, bronchiolitis obliterans |
| Lung nodule | 5–30 mm, malignancy risk model | 1–5 mm micronodules; metastasis screening; many benign lymphoid aggregates / intrapulmonary nodes |
| Cardiomegaly | Chamber enlargement, adult HF | Congenital heart disease morphology; peds reports rarely use the word |
| Lymphadenopathy | ≥10 mm short axis | Peds thresholds vary; "prominent" nodes often normal — only 28/72 mentions reach the impression |

If you keep the adult head weights for these, calibration will be wrong even when
detection works. Re-label, don't just fine-tune.

---

## 6. Prevalence shift within pediatrics (% of reports per age band)

n: <6y = 189, 6–11y = 145, 12–17y = 297, 18+y = 178 (age parseable in 809/1206 = 67%).

| Label | <6y | 6–11y | 12–17y | 18+y |
|---|---|---|---|---|
| Lung nodule | 35.4 | 50.3 | 58.2 | 51.7 |
| Lung opacity | 31.2 | 24.1 | 24.6 | 25.3 |
| Atelectasis | 29.1 | 7.6 | 10.8 | 14.0 |
| Peribronchial thickening | 16.4 | 19.3 | 7.1 | 12.9 |
| Mosaic attenuation | 19.6 | 11.0 | 5.7 | 12.4 |
| Bronchiectasis | 6.9 | 10.3 | 4.7 | 14.6 |
| Pulmonary fibrotic sequela | 12.7 | 12.4 | 9.4 | 13.5 |
| Consolidation | 9.0 | 4.1 | 5.7 | 9.6 |
| Pleural effusion | 3.7 | 1.4 | 4.4 | 8.4 |
| Lymphadenopathy | 5.3 | 5.5 | 6.7 | 7.9 |
| Medical material | 9.0 | 4.1 | 4.7 | 5.1 |
| Cardiomegaly | 1.6 | 2.1 | 0.3 | 0.6 |
| Emphysema | 0.5 | 0.7 | 0.3 | 2.8 |
| Pulmonary metastases | 5.8 | 8.3 | 11.8 | 8.4 |
| Mucus plugging | 3.7 | 4.1 | 2.0 | 10.7 |
| Tree-in-bud | 8.5 | 9.7 | 4.7 | 9.0 |
| Post-surgical change | 10.6 | 9.0 | 13.8 | 19.1 |
| Pulmonary cyst | 8.5 | 8.3 | 6.4 | 10.7 |

Atelectasis and mosaic attenuation are ~3× more common under 6y (sedation, small airways).
Nodules climb with age (oncology surveillance cohort). Stratify your validation split by
age or you will hide this.

---

## 7. Label correlation in pediatric data (φ, body-level, labels with N≥60)

|  | MedMat | LymphAd | Atel | Nodule | Opacity | Fibrotic | Peribr | Consol | Bronchiect | Mosaic |
|---|---|---|---|---|---|---|---|---|---|---|
| Medical material | 1.00 | 0.14 | 0.14 | -0.04 | 0.10 | 0.01 | 0.01 | 0.18 | -0.00 | 0.03 |
| Lymphadenopathy | 0.14 | 1.00 | 0.12 | 0.03 | 0.17 | 0.15 | 0.15 | 0.16 | 0.03 | -0.03 |
| Atelectasis | 0.14 | 0.12 | 1.00 | -0.10 | 0.25 | 0.28 | 0.06 | 0.21 | 0.04 | 0.08 |
| Lung nodule | -0.04 | 0.03 | -0.10 | 1.00 | 0.03 | -0.03 | -0.03 | -0.02 | -0.04 | -0.14 |
| Lung opacity | 0.10 | 0.17 | 0.25 | 0.03 | 1.00 | 0.15 | 0.18 | 0.27 | 0.17 | 0.10 |
| Pulmonary fibrotic sequela | 0.01 | 0.15 | 0.28 | -0.03 | 0.15 | 1.00 | 0.19 | 0.04 | 0.24 | 0.09 |
| Peribronchial thickening | 0.01 | 0.15 | 0.06 | -0.03 | 0.18 | 0.19 | 1.00 | 0.07 | 0.37 | 0.20 |
| Consolidation | 0.18 | 0.16 | 0.21 | -0.02 | 0.27 | 0.04 | 0.07 | 1.00 | 0.09 | 0.01 |
| Bronchiectasis | -0.00 | 0.03 | -0.04\* | -0.04 | 0.17 | 0.24 | 0.37 | 0.09 | 1.00 | 0.14 |
| Mosaic attenuation | 0.03 | -0.03 | 0.08 | -0.14 | 0.10 | 0.09 | 0.20 | 0.14 | 0.14 | 1.00 |

\* rounding from the raw matrix (0.04).

Two clusters: an **airways cluster** (peribronchial thickening ↔ bronchiectasis 0.37,
↔ mosaic 0.20 — the CF cohort) and an **opacity cluster** (opacity ↔ consolidation 0.27,
↔ atelectasis 0.25). Lung nodule is nearly orthogonal to everything (max |φ| = 0.14) —
it is a separate axis and deserves its own loss weighting / possibly its own head.

---

## 8. Recommended pediatric label set

**Keep from CT-RATE (12):** Lung nodule, Lung opacity, Atelectasis, Peribronchial
thickening, Pulmonary fibrotic sequela\*, Mosaic attenuation pattern, Bronchiectasis,
Consolidation, Lymphadenopathy, Medical material, Pleural effusion, Interlobular septal
thickening. (\* redefine as post-infectious scarring.)

**Drop (4):** Coronary artery wall calcification, Arterial wall calcification, Hiatal
hernia, Emphysema (or redefine as "bulla / pneumatocele / lobar overinflation").

**Borderline keep (2):** Cardiomegaly, Pericardial effusion — cheap to keep, ~1–2%
prevalence, will be low-AP but clinically actionable.

**Add (9 high-value):** Post-surgical / post-treatment change (12.7%), Pulmonary
metastases (8.5%), Tree-in-bud (7.3%), Pulmonary cyst (7.2%), Mass / neoplasm (6.0%),
Mucus plugging (4.3%), Pleural thickening / nodule (3.8%), Bone lesion / fracture (2.5%),
Pneumothorax (2.0%).

**Add if you can pool more data (rare but peds-defining):** Congenital lung malformation
(CPAM/sequestration), NEHI/ChILD, congenital vascular anomaly, tracheo-/bronchomalacia,
thymic abnormality. Each <1% here — not trainable from 1206 reports alone.

Resulting set: 12 + 2 + 9 = **23 labels**, vs 18 adult. Transfer plan: keep the backbone,
re-init the 4 dropped heads, warm-start the 12 shared heads, cold-start the 9 new ones,
and re-fit thresholds on pediatric data (prevalence differs by 5–20× on several labels).

---

## 9. Caveats

- Labels here are **regex + negation heuristics over report text**, not radiologist
  ground truth. Negation window is ~60 chars back, cut at clause breaks. No hand-labeled
  gold set exists, so precision/recall of this extraction is unmeasured.
- "Pneumonia/infection" style hedging ("likely infectious or inflammatory etiology") is
  counted as positive; treat infection-adjacent counts as upper bounds.
- Age parsed from the INDICATION line only — 397/1206 (32.9%) have no parseable age.
- CT-RATE label definitions were re-implemented from label names, not from the official
  CT-RATE annotation protocol. If you have that protocol, re-run §2 with its exact
  criteria before freezing the label set.
- Reports are one per study; repeat studies on the same patient are not de-duplicated,
  so prevalence is per-study, not per-patient (surveillance cohorts inflate nodule %).

Generated from: `just_reports.xlsx` (1206 rows) via `classify_reports.py` and
`ctrate_map.py`. Per-report binary label matrix: `ctrate_labels_pediatric.csv`.
