# RSNA Knee Abnormality Detection: learning a 12-finding knee-MRI model from radiology reports

An end-to-end solution to the Kaggle **RSNA Knee Abnormality Detection** challenge (12 findings per knee MRI study, metric = macro-averaged AUC).
Only 58 of the 4,407 training studies have real labels. Everything else comes with a free-text, multilingual radiology report. This project turns those
reports into training labels with an LLM, trains a multi-series MRI model on them, and ships an offline Kaggle inference notebook.

| | Score |
|---|---|
| **Kaggle private leaderboard** | **0.886** |
| **Kaggle public leaderboard** | **0.895** |
| Internal check: 58 gold studies, one model (flip, fold 0) | 0.877 macro AUC |
| Internal check: 58 gold studies, same model without laterality handling | 0.842 macro AUC |

Leaderboard rank / late-submission status: **[ADD]**.
For context only: the public notebooks I studied reported higher public scores (a CoAtNet checkpoint at 0.924, a blend of public checkpoints at 0.958).
**None of their weights, code or notebooks are used here**; the goal was an original, understood pipeline. Gold-set numbers are on n = 58 and are noisy
(about ±0.03 on the macro, ±0.1 on a single label); see [Limitations](#9-limitations-and-honest-caveats).

---

## Contents
1. [Problem and data](#1-problem-and-data)
2. [Approach at a glance](#2-approach-at-a-glance)
3. [Step-by-step: what was done](#3-step-by-step-what-was-done)
4. [Results](#4-results)
5. [Notebooks and links](#5-notebooks-and-links)
6. [Prompts used with Qwen](#6-prompts-used-with-qwen)
7. [Debugging story and lessons](#7-debugging-story-and-lessons)
8. [Reproducing the pipeline](#8-reproducing-the-pipeline)
9. [Limitations and honest caveats](#9-limitations-and-honest-caveats)
10. [Repository layout](#10-repository-layout)
11. [Future work](#11-future-work)

---

## 1. Problem and data

- **Task:** for each knee MRI study, predict a probability for 12 findings: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA,
  Effusion, Synovitis, Baker's cyst, Contusion, Fracture. Score = mean of the 12 per-finding ROC AUCs. Code competition: offline notebook, GPU/CPU ≤ 9 h.
- **Training data:** 4,407 studies, 24,371 series (about 5.5 series per study, median 30 slices per series), DICOM. Each study has a `Report` in one of
  about ten languages (English, Spanish, Turkish, Croatian/Bosnian, Greek, German, Bulgarian/Cyrillic, Dutch, French, ...).
- **Labels:** only **58** studies carry the 12 binary labels ("gold"). The other 4,349 have a report and nothing else.
- **Test data:** about 1,300 studies, images only (no reports), hidden.

Key facts from the EDA (`rsna_eda.py`):

| Finding | Number | Consequence |
|---|---|---|
| `Fluid_Sensitive` vs `Fat_Suppression` | identical in every train series (phi = 1.00) | treat as one "fat-suppressed fluid-sensitive" flag, keep both as inputs |
| Plane label vs DICOM geometry | 0% mismatch | plane is derived from geometry anyway |
| FOV (p5 / p50 / p95) | 140 / 160 / 189 mm; pixel spacing 0.17-0.56 mm | crop a fixed physical box, not a fixed pixel box |
| Series with > 150 slices | 314 (max 320) | physical-position slice sampling and slab averaging |
| Intensity (p99) range across series | 321x | per-series normalization is mandatory |
| Transfer syntax in train | 100% Explicit VR Little Endian | test is said to be mixed: decoder tests added |
| Gold prevalence | high (e.g. Effusion 60%, Medial Meniscus 45%); mean 4.14 positives per study; no all-negative study | gold is abnormal-enriched: use it for ranking, not calibration |
| Reports | 183 studies share an identical report text; median length about 1,000 characters | templated reports are unreliable |

---

## 2. Approach at a glance

```
 reports (multilingual) ──Stage 0 hygiene──► clean text + side + baseline states
        │
        └─Stage 1: Qwen3-8B (vLLM) structured extraction ─► per-finding state / severity / qualifier / evidence
                                  │
                                  └─Stage 2 label model (fit on the 58 gold) ─► silver probabilities p_<label> + reliability weight w
 DICOMs ──dicom_cache──► canonical uint8 cache (series selection, orientation, physical crop, slabs)
 side (rules + DICOM tag + description + image CNN) ─► laterality canonicalisation
 group folds by site proxy, gold held out ─► train 2.5D multi-series model on silver ─► 2-seed rank-averaged ensemble ─► offline inference
```

The design rule that shaped everything: **the LLM records what the report says; it does not decide the label.** The annotators' conventions (for example "a small
Baker's cyst is 0", "grade-1 sprain is 0") are not stated in the reports, so they are learned afterwards from the 58 gold studies by a small regularised model.

---

## 3. Step-by-step: what was done

### 3.1 EDA (`rsna_eda.py`)
A one-pass audit of the CSVs, the DICOM headers and a sample of pixels: integrity (ids, folders, slice counts), label prevalence with Wilson intervals and
AUC resolution, report language and duplicates, plane x fat-suppression structure, header geometry, decodability per transfer syntax, a montage of central
slices, and a "pseudo-site" grouping (manufacturer, model, field strength, language) used later for validation folds.
**Notebook: https://www.kaggle.com/code/khushvaghela/rsna-eda**

### 3.2 Stage 0: data hygiene (`stage0_hygiene.py`)
- **Encoding repair** (UTF-8 bytes read as Windows-1252, including dropped bytes resolved from corpus vocabulary). The real mojibake rate was only **0.1%**;
  garbled text I had seen earlier was a spreadsheet-viewer artifact, not corruption in the files.
- **Normalisation and language ID** (script + stop-word heuristic). Fixes found during the audit: a Greek header character (`∆` U+2206 instead of `Δ`), Latin
  look-alike letters inside Greek words, Dutch reports misclassified as Spanish, and `[BILATERAL NOTE]` multi-report studies.
- **Knee side from the report** (multilingual regex, including Greek abbreviations `ΔΕ` / `ΑΡ`).
- **Rule-based per-finding states** (present / uncertain / absent / not_mentioned) with negation and uncertainty cues: macro AUC vs gold **0.674**. This is the floor
  the LLM extractor has to beat.
- Reading the 58 gold reports next to their labels showed that **gold follows stricter, partly image-based conventions than the report text**: about 10 of 58
  studies contradict their report on a strong statement (my manual reading, not a computed statistic). Examples: explicit "synovitis" labelled 0; a stated meniscal tear labelled 0; "normal" report with
  Baker's = 1. This is why a thresholded mapping is learned later instead of trusting mentions.

### 3.3 Stage 1: structured extraction with Qwen (`stage1_extraction.py`, `stage1_run.py`)
- **Model:** Qwen3-8B (**[confirm exact checkpoint]**), served with **vLLM on Kaggle T4 x2** (tensor parallel 2, fp16, constrained JSON decoding). Qwen3-14B was too
  slow to use; the 8B model through Hugging Face 4-bit generation ran at 58.6 s/study, vLLM at **7.4 s/study** including retries (estimated about 9 h for all 4,407 studies).
- **Output per finding:** `state`, `severity` (the report's own size word), `qualifier` (closed vocabulary such as `grade1_sprain`, `full_thickness_defect_or_ulcer`),
  `size_mm`, a **verbatim evidence quote**, and a `conflict` flag. Plus `report_kind` and `knee_side`.
- **Validation of the output:** JSON parse, enum repair, evidence checked as a substring of the report at four levels (exact, accent/punctuation-insensitive,
  fuzzy, bad). Only structural failures are retried; evidence mismatches are logged (the non-verbatim rate fell from 31% to 10.5% after making the check
  accent-insensitive).
- **Result on the gold studies** (state score only, before the label model): macro AUC **0.806** vs **0.673** for the rule baseline.
- **Bug found:** the LLM's `knee_side` disagreed with every other source in **46.9%** of Turkish reports (the word "sol" = left was read as "right"). The LLM side is
  therefore never used as a label; it does not affect the finding extraction.
- The full prompt is in [Section 6](#6-prompts-used-with-qwen). **Notebook(See Version 15): https://www.kaggle.com/code/khushvaghela29/stage-1**

### 3.4 Stage 2: label model (`stage2_label_model.py`)
Per finding, three small, strongly regularised logistic models are compared with repeated 5-fold CV on the 58 gold studies:
**V0** = calibrated plain state score; **V1** = the finding's own features (state, severity, qualifier, size); **V2** = V1 plus related findings (e.g. Effusion for
Synovitis, Fracture for Contusion). A learned variant replaces V0 only if it beats the raw score by at least 0.03 CV AUC.
- **Chosen:** V1 for ACL, MCL, Medial Meniscus, Effusion; V2 for Fracture; V0 for the other seven.
- **CV AUC on gold:** ACL 0.851 to 0.956, MCL 0.896 to 0.948, Fracture 0.791 to 0.870, Effusion 0.652 to 0.714; macro 0.806 to **0.828** (optimistic: selected and
  scored on the same 58 studies).
- **Outputs:** `silver_labels.csv` with 12 probabilities per study (gold studies get cross-fitted values) and a reliability weight `w`, lowered for
  header-only or near-empty reports, templated duplicates, multi-report studies and weak evidence.
- Silver prevalence on non-gold studies is below the gold prevalence (as expected for an enriched gold set); only rankings are used.

### 3.5 Knee side (`side_sources.py`, `side_classifier.py`)
The image model needs the side (left/right) of every study to put the lateral side in a consistent place.
- **Sources:** report rules (43%), DICOM `Laterality` tag (49%), `SeriesDescription` tokens (13%), LLM (45%, discarded as a label because of the Turkish bug).
  Rules, tag and description agree with each other about 98% of the time. **3,115 of 4,407 studies (71%)** have a trusted side.
- **Image classifier:** a small CNN on three coronal slices, trained on the trusted studies: **0.989 accuracy / AUC 0.993** in 5-fold CV. It supplies the side for the other
  1,315 studies and overrides 23 trusted labels it contradicts at >= 95% confidence (mostly templated reports from one site). On the 28 strong-source conflicts it
  sided with the DICOM tag 67% of the time and the report text 36%.
- **Unresolved diagnostic:** after flipping left knees, the classifier should not be able to tell sides, but it still could (within-site AUC 0.86). My
  hypothesis (not proven) is that flipping turns fixed hardware asymmetries (coil sensitivity, ghosting) into a side-specific cue. The decision was therefore made empirically (3.9).

### 3.6 DICOM cache (`dicom_cache.py`)
One module used for both training and inference so preprocessing cannot drift:
- up to 2 fat-suppressed + 1 non-fat-suppressed series per plane (6 slots = plane x fat-sat; second fat-suppressed series chosen by TE difference);
- 16 slices per series, evenly spaced by physical position over the central 6-94% of the stack; stacks thinner than 1.75 mm are slab-averaged;
- physical square crop of 150 mm around the FOV centre, resized to 224 x 224;
- orientation canonicalised from `ImageOrientationPatient` (patient-based frame; plane taken from geometry);
- per-series percentile windowing to uint8; laterality is applied later (`apply_laterality`).
- **Tests:** a synthetic-DICOM suite checks that 13 orientation variants land a marker on the same pixel, physical scaling is correct at several pixel spacings, and
  Implicit VR / RLE / JPEG 2000 Lossless series decode to identical arrays. JPEG Lossless could not be tested (no encoder available).
- **Cost:** 0.63 s/study with 4 workers, about 12-13 GB for the whole train set (compressed `.npz`). **Notebook: https://www.kaggle.com/code/khushvaghela29/stage-3**

### 3.7 Validation folds (`make_folds.py`)
Group K-fold (k = 5) by a site proxy (normalised manufacturer + model + field strength + language), 48 groups, fold sizes 868 to 877. The **58 gold studies are held out
(fold -1) and never trained on**. Validation on a held-out fold therefore measures generalisation to unseen sites, which is pessimistic.

### 3.8 Model and training (`train_knee.py`)
- **Per study:** all cached series, 6 random windows per series (3 adjacent slices as RGB) for training, 8 evenly spaced for evaluation.
- **Network:** ImageNet-pretrained `convnext_nano` backbone per window, then a head. The final head is a **MIL head**: every window predicts all 12 findings and a learned
  per-finding softmax pools windows. An optional transformer over window tokens (slot, rank and relative-position embeddings) and a per-finding cross-attention head are
  also implemented; the final models use the MIL head without the transformer.
- **Loss:** BCE on the silver probabilities (soft targets) weighted by study reliability `w`, **plus an auxiliary per-window loss** (each window predicts the study target, weight 0.3).
  The output bias is initialised to each label's mean.
- **Optimisation:** AdamW (backbone 1e-4, heads 1e-3), weight decay 0.05, cosine schedule with warm-up, fp16 autocast, EMA of weights (0.999), gradient clipping,
  skip-on-NaN guard. Augmentation: small affine and intensity jitter, series dropout 0.1; **no flips** (they would swap medial and lateral).
- **Speed:** about 5.2 min/epoch on one T4 once data loading was fixed (3,481 training studies, 870 steps, about 0.4 s/step compute).

### 3.9 Frozen-feature probe and the laterality ablation
- **Probe** (`probe_features.py`): frozen ConvNeXt-nano features pooled per slot + ridge regression: **gold AUC 0.790**, site-holdout Spearman 0.30, random-split Spearman 0.43
  (random > site-holdout shows part of what is learnable follows site/report style).
- **Laterality ablation** (same fold, same seed, 8 epochs): flip beat none at every epoch on validation Spearman and gold AUC; see [Results](#4-results). **Decision: flip.**

### 3.10 Final models and inference (`infer_knee.py`)
- **Final models:** laterality flip, MIL head, aux loss 0.3, trained on **all non-gold studies**, two seeds (**[confirm epochs; 10 recommended]**), checkpoints = last epoch (EMA weights).
- **Inference:** per study, preprocessing on the fly with `dicom_cache`, side predicted by the CNN from the coronal images, laterality applied exactly as in training, both
  checkpoints run, predictions **rank-averaged** per label. Failed studies get the per-label median, not a constant. Series-level decode errors and codec availability are logged.
- **Verification:** running the offline code on the 58 gold studies reproduces the training logs (flip **0.8773** vs 0.8770; none **0.8416** vs 0.8416). About 1.2 minutes for 58 studies.
- **Submitted** as an offline Kaggle notebook (internet off, DICOM codec wheels attached as a dataset). **Notebook: https://www.kaggle.com/code/khushvaghela/rsna-submission**

---

## 4. Results

### 4.1 Kaggle
| | Score |
|---|---|
| Public leaderboard | 0.895 |
| Private leaderboard | 0.886 |

### 4.2 How each stage moved the (noisy) gold-58 metric
| Method | Gold macro AUC | Notes |
|---|---|---|
| Rule-based report states | 0.674 | Stage 0 baseline |
| Qwen extraction, plain state score | 0.806 | before the label model |
| + label model (CV, optimistic) | 0.828 | selected on the same 58 studies |
| Frozen ConvNeXt features + ridge | 0.790 | images only, no training |
| Fine-tuned, 2 epochs (MIL + aux / attn + aux) | 0.686 / 0.669 | first working runs |
| Fine-tuned, 8 epochs, no laterality handling | 0.842 | fold 0, one seed |
| Fine-tuned, 8 epochs, laterality flip | **0.877** | fold 0, one seed |
| Final 2-seed ensemble, all non-gold data | **[ADD gold-check value]** | `infer_knee.py --check-gold` |

### 4.3 Flip vs none per finding (gold AUC, fold-0 models; flip = the epoch-6 log line, none = epoch 7)
| Finding | none | flip |
|---|---|---|
| ACL | 0.963 | 0.958 |
| MCL | 0.649 | 0.687 |
| Medial Meniscus | 0.841 | **0.928** |
| Lateral Meniscus | 0.727 | **0.815** |
| Medial OA | 0.960 | 0.978 |
| Lateral OA | 0.814 | 0.868 |
| PF OA | 0.762 | 0.797 |
| Effusion | 0.971 | 0.970 |
| Synovitis | 0.650 | 0.695 |
| Baker's | 0.960 | 0.980 |
| Contusion | 0.876 | 0.924 |
| Fracture | 0.926 | 0.922 |
| **Macro** | **0.842** | **0.877** |

Validation Spearman vs silver (868 held-out studies): flip about 0.43, none 0.41; on the five medial/lateral labels (MCL, both menisci, both OA) 0.446 vs 0.433. Weakest
labels in every run: **Synovitis and MCL** (reports explain them worst), then Lateral Meniscus and PF OA.

---

## 5. Notebooks and links

| Step | Notebook / dataset | Link |
|---|---|---|
| EDA | `rsna_eda` | **https://www.kaggle.com/code/khushvaghela/rsna-eda** |
| Stage 0 hygiene | `stage0_hygiene` | **https://www.kaggle.com/code/khushvaghela/stage-0** |
| Stage 1 extraction (Qwen + vLLM, T4 x2) | `stage1_run` | **https://www.kaggle.com/code/khushvaghela29/stage-1**(See Version 15) |
| Stage 2 label model | `stage2_label_model` | **https://www.kaggle.com/code/khushvaghela29/stage-2** |
| Side sources + side classifier | `side_sources`, `side_classifier` | **https://www.kaggle.com/code/khushvaghela29/side-sources** |
| DICOM cache build | `dicom_cache` | **https://www.kaggle.com/code/khushvaghela29/stage-3** |
| Folds, probe, training runs (ablation, final models) | `make_folds`, `probe_features`, `train_knee` | **https://www.kaggle.com/code/khushvaghela29/fold-and-train** |
| Submission notebook (offline inference) | `infer_knee` | **https://www.kaggle.com/code/khushvaghela/rsna-submission** |
| Scripts dataset | all scripts + side model + final checkpoints | **https://www.kaggle.com/datasets/khushvaghela/rsna-dataset** |
| DICOM codec wheels dataset | `pydicom`, `pylibjpeg`, ... wheels | **https://www.kaggle.com/datasets/khushvaghela/rsna-wheels** |
| Cache dataset | `rsna-cache` | **https://www.kaggle.com/datasets/khushvaghela29/rsna-cache** |
| Silver labels / stage outputs dataset | `side_final.csv` | **https://www.kaggle.com/datasets/khushvaghela29/side-out** |

---

## 6. Prompts used with Qwen

The extraction call is a chat with a **system message**, **two few-shot demonstrations**, and the **target report**. vLLM's constrained decoding enforces the JSON schema; the
validator enforces enums and verbatim evidence.

### 6.1 System message
The system message is the rules below, then the line `FINDINGS AND ALLOWED QUALIFIERS:`, then the per-finding block of 6.2.

```text
You extract structured findings from a single knee-MRI radiology report. Reports may be in any language
(English, Spanish, Portuguese, French, German, Dutch, Italian, Turkish, Greek, Bulgarian, Croatian/Bosnian, ...).
Read the report in its original language. Do not translate the evidence.

You RECORD what the report states. You do NOT decide diagnoses and you do NOT apply any threshold of your own.
Never infer a finding from the clinical history/indication alone ("Antecedentes: Artrosis", "suspected meniscal tear" in
the indication are not findings). Use the findings and conclusion/impression sections.

For each of the 12 findings give:
  state      present | absent | uncertain | not_mentioned
             present  = the report states the finding (also "compatible with", "consistent with", "in favor of", "C/W").
             uncertain = "possible", "suspected", "cannot be excluded", "may represent", "R/O", "query", "probably" -- the report itself is unsure.
             absent   = the report explicitly negates it OR calls the relevant structure normal/intact/unremarkable
                        (a collective statement such as "cruciate and collateral ligaments within normal limits" is absent for ACL and MCL;
                        quote that phrase as evidence for each).
             not_mentioned = the report says nothing about it. Silence is NOT absence.
  severity   the report's own size/degree word: na | minimal | mild | moderate | marked | complete | unspecified
             (trace/minimal/ligera -> minimal or mild per the word used; massive/large/severe/complete -> marked or complete).
  qualifier  one value from the allowed list for that finding (type or grade as the report states it).
  size_mm    largest measured dimension in millimetres if the report gives numbers (cm -> mm), else omit.
  evidence   ONE short quote copied EXACTLY (character for character, same language) from the report that supports the state.
             Maximum 25 words. Omit for not_mentioned. Never paraphrase or translate evidence.
  conflict   true only if the report contradicts itself about this finding (e.g. "no fracture" in the findings but a fracture in the impression).
Omit fields that do not apply; for not_mentioned give only {"state": "not_mentioned"}.

Global fields:
  report_kind   full_structured | brief_summary (a few lines) | generic_normal_template (boilerplate normal text that does not describe a
                specific patient's abnormalities) | multiple_reports (more than one report/knee in the text) | other
  knee_side     left | right | both | unknown   (only if the report says it)
  knee_side_evidence  exact quote if side is given.

Compartments: medial = internal; lateral = external; patellofemoral = patella + trochlea. Meniscus/ligament/cartilage statements
belong to the side they name. If a report mentions two reports/knees, set report_kind=multiple_reports and extract from the whole text.
Return ONLY a JSON object, no commentary.
```

### 6.2 Per-finding definitions and closed qualifier vocabularies
```text
- "ACL": Anterior cruciate ligament (ACL / LCA / VKB / LCA / ön çapraz bağ / πρόσθιος χιαστός / предна кръстна връзка). A statement about 'cruciate ligaments' in general applies to ACL (and PCL).
    qualifier in: complete_or_rupture, high_grade_partial, partial_or_interstitial, grade1_or_sprain, mucoid_or_degenerative, postoperative, not_specified
- "MCL": Medial (tibial) collateral ligament only. Lateral/fibular collateral ligament is NOT MCL. A statement about 'collateral ligaments' in general applies to MCL.
    qualifier in: complete, partial_grade2, grade1_sprain, thickening_or_edema_only, not_specified
- "Medial Meniscus": Medial meniscus (= internal, 'menisco interno', Innenmeniskus, έσω μηνίσκος, medijalni, медиален, iç/medial menisküs, mediaal).
    qualifier in: tear, complex_bucket_handle_or_root, degeneration_only, postoperative_meniscectomy, extrusion_only, signal_change_unclear, not_specified
- "Lateral Meniscus": Lateral meniscus (= external, 'menisco externo', Außenmeniskus, έξω μηνίσκος, lateralni, латерален, dış/lateral menisküs, laterale).
    qualifier in: tear, complex_bucket_handle_or_root, degeneration_only, postoperative_meniscectomy, extrusion_only, signal_change_unclear, not_specified
- "Medial OA": Cartilage loss / osteoarthritis of the MEDIAL femorotibial compartment (medial femoral condyle + medial tibial plateau). Record any cartilage statement for that compartment, even if the word osteoarthritis is not used.
    qualifier in: osteoarthritis_stated, full_thickness_defect_or_ulcer, partial_thickness_or_fissuring, superficial_or_low_grade, narrowing_or_osteophytes_only, not_specified
- "Lateral OA": Cartilage loss / osteoarthritis of the LATERAL femorotibial compartment (lateral femoral condyle + lateral tibial plateau).
    qualifier in: osteoarthritis_stated, full_thickness_defect_or_ulcer, partial_thickness_or_fissuring, superficial_or_low_grade, narrowing_or_osteophytes_only, not_specified
- "PF OA": Cartilage loss / osteoarthritis of the PATELLOFEMORAL compartment (patella, trochlea; 'condropatía rotuliana', chondromalacia patellae).
    qualifier in: osteoarthritis_stated, full_thickness_defect_or_ulcer, partial_thickness_or_fissuring, superficial_or_low_grade, narrowing_or_osteophytes_only, not_specified
- "Effusion": Joint effusion / excess intra-articular fluid. Put the size word in `severity` (trace/minimal, small/mild, moderate, large/massive). Fluid only in bursae, around a cyst or in muscle is not joint effusion.
    qualifier in: hemarthrosis_or_hemorrhagic, simple, not_specified
- "Synovitis": Synovitis / synovial thickening, hypertrophy or proliferation / Hoffa fat-pad inflammation or impingement. Put what the report literally says in `qualifier`.
    qualifier in: explicit_synovitis, synovial_thickening_or_hypertrophy, hoffa_fat_pad_inflammation, bursitis_only, not_specified
- "Baker's": Baker's / popliteal (semimembranosus-gastrocnemius) cyst. Put size words in `severity` and the largest measured dimension in `size_mm`.
    qualifier in: simple, ruptured_or_leaking, not_specified
- "Contusion": Bone contusion / bone bruise / traumatic bone-marrow oedema. Marrow oedema explained as degenerative/subchondral next to cartilage loss is 'marrow_edema_only'. Muscle or soft-tissue contusion is NOT bone contusion.
    qualifier in: bone_contusion_stated, marrow_edema_only, osteochondral_impaction, muscle_or_soft_tissue_only, not_specified
- "Fracture": Any fracture: fracture, subchondral / insufficiency / stress fracture, impaction, avulsion, osteochondral fracture.
    qualifier in: fracture, subchondral_or_impaction, insufficiency_or_stress, avulsion, osteochondral, suspected_or_possible, not_specified
```

### 6.3 Output schema (enforced by constrained decoding)
```text
{ "report_kind": one of [full_structured, brief_summary, generic_normal_template, multiple_reports, other],
  "knee_side": one of [left, right, both, unknown],
  "knee_side_evidence": string,
  "findings": { "<each of the 12 findings>": { "state": required, one of [present, absent, uncertain, not_mentioned],
                                               "severity": one of [na, minimal, mild, moderate, marked, complete, unspecified],
                                               "qualifier": string (checked against the finding's list afterwards),
                                               "size_mm": number, "evidence": string (max 300 chars), "conflict": boolean } } }
```

### 6.4 Few-shot demonstrations (user / assistant turns)
Demo 1 (English), user turn = `REPORT:` + text:
```text
MRI right knee. Findings: Complete tear of the anterior cruciate ligament. Vertical tear of the posterior horn of the medial meniscus. The lateral meniscus is normal. Grade 1 sprain of the medial collateral ligament. Small joint effusion. Baker's cyst measuring 32 x 15 mm. No fracture. Impression: ACL rupture, medial meniscus tear.
```
Assistant turn:
```json
{"report_kind": "full_structured", "knee_side": "right", "knee_side_evidence": "MRI right knee",
 "findings": {
  "ACL": {"state": "present", "severity": "complete", "qualifier": "complete_or_rupture", "evidence": "Complete tear of the anterior cruciate ligament"},
  "MCL": {"state": "present", "severity": "mild", "qualifier": "grade1_sprain", "evidence": "Grade 1 sprain of the medial collateral ligament"},
  "Medial Meniscus": {"state": "present", "severity": "unspecified", "qualifier": "tear", "evidence": "Vertical tear of the posterior horn of the medial meniscus"},
  "Lateral Meniscus": {"state": "absent", "evidence": "The lateral meniscus is normal"},
  "Medial OA": {"state": "not_mentioned"}, "Lateral OA": {"state": "not_mentioned"}, "PF OA": {"state": "not_mentioned"},
  "Effusion": {"state": "present", "severity": "mild", "qualifier": "simple", "evidence": "Small joint effusion"},
  "Synovitis": {"state": "not_mentioned"},
  "Baker's": {"state": "present", "severity": "unspecified", "qualifier": "simple", "size_mm": 32, "evidence": "Baker's cyst measuring 32 x 15 mm"},
  "Contusion": {"state": "not_mentioned"},
  "Fracture": {"state": "absent", "evidence": "No fracture"}}}
```
Demo 2 (Spanish): the report text is
```text
Antecedentes Clínicos: Artrosis. Hallazgos: Ligamentos cruzados y colaterales dentro de límites normales. Menisco medial con aumento de señal sin rotura. Úlcera condral de espesor total del cóndilo femoral lateral. Leve derrame articular con mínima sinovitis. No hay quistes poplíteos. Impresión: Condropatía grado 4 del compartimento femorotibial lateral.
```
and the expected output shows the key behaviours the rules ask for: the clinical-history word "Artrosis" is **not** extracted as a finding; a collective "normal" statement
marks both ACL and MCL `absent` with the same quote; "aumento de señal sin rotura" gives Medial Meniscus `absent` with qualifier `signal_change_unclear`; the full-thickness chondral
ulcer is Lateral OA `present` / `full_thickness_defect_or_ulcer`; "mínima sinovitis" is Synovitis `present`/`minimal`; Baker's is `absent`; everything unmentioned is `not_mentioned`.

### 6.5 Retry message (only for structural failures)
If the output is not valid JSON, misses a finding, or has an invalid enum, the conversation is repeated with the bad answer and this follow-up:
```text
Your previous JSON had problems: <list of issues>. Return the COMPLETE corrected JSON object again. Every `evidence` must be copied character-for-character from the report (same language, no translation, no paraphrase, <= 25 words); if you cannot quote it exactly, use a shorter exact phrase. Use only the allowed enum values. Output JSON only.
```

### 6.6 Optional compact variant (`--compact`)
Shortens evidence to 12 words and asks for single-line JSON; it changes three phrases of the system message (`Maximum 25 words.` to `Maximum 12 words.`, an added "keep it as short as possible" sentence for `evidence`, and a single-line-JSON instruction). **[State whether the final run used it.]**

### 6.7 Prompt design notes
- Record, do not decide: severity and qualifier words are extracted verbatim-style; thresholds are learned later (Stage 2).
- Four states, not three: "not mentioned" is kept apart from "absent", because silence in a report is common and is not a negative.
- Verbatim evidence plus a programmatic substring check makes hallucinated evidence detectable.
- Medial/lateral synonym lists in ten languages are inside the definitions.
- The side field exists but is **not trusted** (Turkish bug); side comes from rules, DICOM tags and the image classifier.

---

## 7. Debugging story and lessons

1. **Encoding "corruption" that wasn't.** Garbled Cyrillic/Greek/Spanish text led to a repair module; the actual mojibake rate was 0.1%. The garbling was a viewer artifact.
   Lesson: measure on the real file before building a fix.
2. **Report-vs-gold mismatch.** The labels follow conventions the reports do not state, which led to the "extract, then learn the thresholds" design.
3. **Qwen3-14B too slow, then vLLM.** 4-bit HF generation was 58.6 s/study; vLLM with tensor parallelism and constrained decoding was 7.4 s/study.
4. **LLM side bug.** The extractor mis-read Turkish "sol" as right in about half of Turkish studies. Cross-checking against independent sources found it.
5. **Training stuck at the label means.** The first 8-epoch runs ended with train loss 0.5444 against a constant-prediction floor of 0.5436 and gold AUC about 0.5.
   - Data loading was one cause of slowness: cold reads from the dataset mount cost 0.642 s/study vs 0.026 s warm. Copying the cache to local disk fixed it.
   - An overfit test (96 studies, no augmentation) learned slowly (loss 0.4977), so the pipeline was sound but optimisation was weak.
   - My first retry commands used `--lr 1e-3`, which the script multiplies by 10 for the heads, causing fp16 NaNs; the logging average also hid the cause. Fixes: explicit `--head-lr-mult`, NaN-skip guard.
   - What fixed learning: a **per-window auxiliary loss**, a **MIL head** with direct gradients to each window, and **bias initialised to the label means**. Two epochs then gave val Spearman 0.27 and gold AUC about 0.69.
6. **A frozen-feature probe was the most informative cheap experiment**: it showed signal existed (gold AUC 0.79) before the fine-tuning problem was solved.
7. **Laterality:** the flip diagnostic was ambiguous, so the choice was made by an ablation on the real objective. Flip won on gold AUC (+0.035), with the largest gains on the meniscus labels.
8. **Inference verification:** reproducing the training-time gold AUC through the offline code path (0.8773 vs 0.8770) caught nothing, which is the point of the check.
9. **Process lessons:** decide with the metric you care about; keep an "honest set" (gold) out of training; make failures visible in logs (series decode errors, per-step timing); keep one preprocessing module for train and test.

---

## 8. Reproducing the pipeline

Order of execution (each script has `--help`, a self-test where noted, and can be called from a notebook as `main([...])`):

```text
1.  python rsna_eda.py            --data-root <comp> --out eda_out
2.  python stage0_hygiene.py      --train-csv <comp>/train.csv --out stage0_out                      (--self-test available)
3.  python stage1_run.py          --reports stage0_out/reports_clean.csv --out s1_full --backend vllm --model <Qwen3-8B dir> --quant none
                                  --chunk 200 --max-new-tokens 1800 --time-budget-min 600            (resumable; Kaggle T4 x2)
    python stage1_run.py --merge  --reports stage0_out/reports_clean.csv --out s1_full
4.  python stage2_label_model.py  --features s1_full/stage1_features.csv --reports stage0_out/reports_clean.csv --out s2_full
5.  python dicom_cache.py         --data-root <comp> --split train --out cache --workers 4         (--self-test available)
6.  python side_sources.py scan   --data-root <comp> --out side_out
    python side_sources.py combine --reports ... --features s1_full/stage1_features.csv --dicom side_out/side_dicom_study.csv --out side_out
    python side_classifier.py     --cache cache --side-table side_out/side_table.csv --out side_clf --arch cnn
    python side_classifier.py     --cache cache --side-table side_out/side_table.csv --out side_clf2 --arch cnn --save-model side_model.pt --final-only
7.  python make_folds.py          --reports ... --headers header_sample.csv --silver s2_full/silver_labels.csv --out folds.csv
8.  python probe_features.py      --cache cache --silver ... --folds folds.csv --reports ... --out probe_out                 (optional probe)
9.  python train_knee.py --fold 0 --laterality {flip,none} --head mil --layers 0 --aux-window 0.3 --epochs 8 ...         (ablation)
10. python train_knee.py --fold -1 --seed {0,1} --laterality flip --head mil --layers 0 --aux-window 0.3 --epochs 10 ...  (final models)
11. python infer_knee.py --data-root <comp> --split train --reports ... --check-gold --ckpts a.pt,b.pt --side-model side_model.pt   (verification)
    python infer_knee.py --data-root <comp> --split test  --ckpts a.pt,b.pt --side-model side_model.pt --out /kaggle/working        (submission)
```
Practical notes: copy the cache from the Kaggle dataset mount to local disk before training (cold reads are slow); train one fold per GPU in parallel; the submission notebook needs
the codec wheels installed offline. `torch`/`timm` code paths have a `--smoke` mode for a quick end-to-end check.

---

## 9. Limitations and honest caveats

- **Tiny, noisy gold set.** 58 studies give about ±0.03 on the macro AUC and about ±0.1 on a single label. Model-selection decisions (label-model variants, laterality, number of epochs) looked at it,
  so gold numbers are slightly optimistic. The Kaggle leaderboard is the independent estimate.
- **Report-derived labels are a proxy.** About 10 of 58 gold studies contradict their report on a strong statement, and gold seems to follow annotator conventions that reports do not state.
  Synovitis and MCL are the weakest findings in every run.
- **Single fold, single seed ablations.** The laterality gain (+0.035 gold AUC) is consistent across epochs and findings but is not a multi-seed result.
- **Unexplained diagnostic.** The post-flip side-leakage explanation (hardware cues) is a hypothesis.
- **Codec coverage.** Implicit VR, RLE and JPEG 2000 Lossless were tested on synthetic files; JPEG Lossless was not. The example test studies were all Explicit VR Little Endian.
- **No label-refinement loop** was run, and no ablation of the final ensemble beyond the leaderboard score.
- The date of submission relative to the competition deadline and the resulting leaderboard status: **[ADD]**.

---

## 10. Repository layout

| File | Purpose |
|---|---|
| `rsna_eda.py` | dataset audit (integrity, labels, reports, series structure, headers, decoding, pseudo-sites) |
| `stage0_hygiene.py` | encoding repair, language ID, side from text, rule-based finding states, gold comparison |
| `stage1_extraction.py` | extraction prompt, schema, validator (verbatim evidence levels), feature encoder |
| `stage1_run.py` | batch runner (HF / vLLM / mock), sharding, resume, retries, merge, gold evaluation |
| `stage2_label_model.py` | per-finding label model fitted on gold; produces silver labels and weights |
| `dicom_cache.py` | DICOM to canonical uint8 cache; shared with inference; synthetic-DICOM self-test |
| `side_sources.py` | side from DICOM tags/descriptions, combined with report side; conflicts |
| `side_classifier.py` | coronal-image side CNN, arbitration, flip diagnostic, final model export |
| `make_folds.py` | site-grouped folds with gold held out |
| `train_knee.py` | the image model (MIL / attention heads, aux loss, EMA, diagnostics) |
| `probe_features.py` | frozen-feature ridge probe |
| `infer_knee.py` | offline inference, rank-averaged ensemble, `submission.csv`, gold check |

---

## 11. Future work

- **Label refinement:** 5-fold out-of-fold image predictions vs silver labels; re-read the studies where they disagree most and refit the label model with the image prediction as a feature.
- **More diverse ensemble members** (different backbone, resolution, or head) and inference-time variants (more windows per series), evaluated on gold and on the leaderboard.
- **Better handling of Synovitis and MCL**, the weakest findings, possibly with image-only supervision signals or per-finding early stopping.
- **Test JPEG Lossless decoding** on real files, and run a multi-seed, multi-fold laterality ablation.
- Resolve the side-leakage diagnostic with a high-pass or cropped version of the flip test.
