# `data_preprocessing.ipynb` — Engineering Audit Report

> A structured review of the project's data-preparation pipeline, audited against 2024-2025 best-practice guidance for fine-grained mushroom image classification and detection. This document is the report; the cell ids it references are the ground truth. Where the report and the code disagree, the code wins — file an issue and update the report.

---

## 1. Executive Summary

`data_preprocessing.ipynb` is a thirteen-step pipeline that turns ~500 mushroom images from public biodiversity APIs (iNaturalist, GBIF, FungiTastic, DF20) into two model-ready datasets: a 224×224 classification set under `data/processed/classification_data_aug/` and a YOLO-style detection set under `data/processed/detection_data/`. The pipeline is **functional, end-to-end reproducible, and idempotent on re-run**, which is more than many notebooks can claim.

The pipeline is also a **prototype** — the audit identifies 30+ concrete issues that fall into four buckets:

| Bucket | Count | Examples |
|---|---:|---|
| 🔴 **Blocker** — produces wrong or unsafe outputs | 2 | Placeholder full-frame YOLO bboxes teach the detector to ignore localization; subset selection biases the train split |
| 🟠 **High** — measurably degrades model quality or developer experience | 9 | `np.random.seed()` global side effect; `except: pass` swallows failures; hue jitter on color-diagnostic species; `metadata.csv` has no schema/version |
| 🟡 **Medium** — notable improvement that should be made in the next iteration | 11 | Aspect-ratio-distorting resize; SHA-256 only dedup; no k-fold; per-call `%pip install`; `_aug` substring filter is fragile |
| 🟢 **Low** — nice-to-have polish | 5 | Filename sanitization; visualization embeds outdated pipeline; inconsistent cell ids; markdown naming convention |

**Overall verdict:** *Working prototype, ready for first training runs but not for production or publication.* The single most important next action is **replacing the placeholder YOLO bboxes in Step 10 with real annotations** — without that, the detection model is a degenerate baseline, not a useful product. Everything else is incremental.

### What this report is, and what it is not

- **Is:** a curated walkthrough of the pipeline's design choices, with each choice tied to a cell id in the notebook and a recommendation for improvement.
- **Is not:** a copy of the notebook's prose. The notebook explains *what each step does*; this report adds the *why it was done this way* and the *what the next contributor should know before changing it*.

---

## 2. Background and Scope

### 2.1 What the pipeline does, in plain English

A mushroom photograph is taken in the wild, uploaded to a citizen-science platform, and labeled with a Latin binomial. By the time it lands in `data/raw/<source>/<Species>/`, the photograph has been seen by humans, voted on, and tagged. `data_preprocessing.ipynb` does six things to it, in order:

1. **Acquire** (Steps 3.1–3.5): pull up to 30 images per (species, source) from four public APIs and place them in `data/raw/<source>/<Species>/<file>.jpg`.
2. **Catalogue** (Step 4): walk `data/raw/` and build a pandas DataFrame with one row per image, tagged with the species and the source dataset.
3. **Filter** (Steps 5–6): keep only the eight target species, cap at 50 per (species, source), drop images smaller than 128 px, and dedupe by SHA-256.
4. **Split** (Step 7): assign each image to `train`, `val`, or `test` with a stratified 70/15/15 split (seed=42).
5. **Reshape** (Steps 8–9): copy every image into `data/processed/classification_data/{split}/{Species}/`, then write a parallel tree of augmented 224×224 JPEGs to `data/processed/classification_data_aug/`.
6. **Branch** (Steps 10–12): copy the same raw images into a YOLO-style `detection_data/` tree with placeholder bboxes, and write a `metadata.csv` summarizing everything.

The output of the pipeline is consumed by `train_classification.ipynb` (which reads `classification_data_aug/`) and `train_detection.ipynb` (which reads `detection_data/`), plus any future notebook that wants the per-image metadata.

### 2.2 Downstream consumers

| Consumer | What it reads | What it expects | What happens if `metadata.csv` is missing or stale |
|---|---|---|---|
| `train_classification.ipynb` | `data/processed/classification_data_aug/{train,val,test}/<Species>/*.jpg` | 224×224 JPEGs, one folder per class | `ImageFolder` raises; the user sees a stack trace and the run aborts |
| `train_detection.ipynb` | `data/processed/detection_data/images/{train,val,test}/*.jpg` + matching `labels/*.txt` | YOLO-convention folder layout with matching image/label stems | Ultralytics raises; the user sees a stack trace and the run aborts |
| A future exploratory notebook | `data/processed/metadata.csv` | columns: `image_path, dataset, raw_species, canonical_species, split, sha256` | `KeyError` on missing columns; the user sees a stack trace |

The "sees a stack trace" pattern is by design — the pipeline is a contract; its consumers trust the directory layout. The cost of that contract is that **any cell in Steps 4–12 can silently produce a `data/processed/` tree that violates the consumers' expectations** (e.g. wrong species folder name, missing split, corrupt JPEG). The audit's high-priority items are about closing those silent-failure windows.

### 2.3 Audit methodology

Each of the thirteen steps is evaluated against four lenses:

1. **Correctness** — does the code produce the right outputs for the documented inputs?
2. **Idempotency** — does re-running the cell produce no spurious changes?
3. **Robustness** — does the code fail loudly (not silently) on bad inputs?
4. **Best-practice fit** — does the implementation match current guidance for image preprocessing in 2024-2025?

Findings are tagged with a severity (🔴 Blocker, 🟠 High, 🟡 Medium, 🟢 Low) and an effort estimate (S = < 1 hour, M = 1–4 hours, L = > 4 hours). Severity reflects *impact on the next experiment*, not theoretical risk.

---

## 3. Pipeline Architecture

The data flow has five stages. Each cell id is in the form `md-N` (markdown) or `code-N` (code) for Steps 1, 2, 4–8, 11–13, and `stepN-M-md` / `stepN-M-code` for Step 3 (which has five sub-fetchers) and Steps 9–10 (which use the `step9-` / `step9-det-` prefix).

```
                 ACQUIRE                CATALOGUE              FILTER
                 ───────                ─────────              ──────
iNaturalist ─┐                                                               
GBIF          │                                                              
FungiTastic   ├──> 3.1-3.5 (md: step3-main-md, code: step3-1/2/3/4/5)   ──>  4. (md-5 / code-6)   ──>  5. (md-7 / code-8)
DF20         │                            4 APIs → data/raw/                  build metadata          cap at 50/(sp,src)
             │                                                              pandas DataFrame
             └─────────────────> data/raw/<src>/<Species>/*.jpg
                                          │
                                          ▼
                                    6. (md-9 / code-10)
                                       validate (>=128 px)
                                       SHA-256 dedup
                                          │
                                          ▼
                 SPLIT                   RESHAPE                    BRANCH
                 ─────                   ───────                    ──────
                                    ┌──> 7. (md-11 / code-12)             ┌──> 10. (step9-det-md / step9-det-code)
                                    │    stratified 70/15/15               │     YOLO format
                                    │    seed=42                           │     placeholder bboxes
                                    │                                      │     yolo.yaml
                                    ▼                                      │
                          8. (md-13 / code-14)                            │
                             shutil.copy2                                 │
                             → classification_data/                       │
                                    │                                      │
                                    ▼                                      │
                          9. (step9-md / step9-code)                       │
                             Albumentations Compose                       │
                             → classification_data_aug/   ─────────────────┤
                             (parallel tree, originals untouched)         │
                                                                              │
                                                                              ▼
                                                              11. (step-8-md / step-8-code)
                                                                              visualize
                                                                              ↓
                                                              12. (md-15 / code-16)
                                                                              metadata.csv
                                                                              ↓
                                                              13. (md-17, markdown only)
                                                                              handoff instructions
```

The classification and detection paths **share no code** but consume the same source images via `metadata.csv`. Step 9 is the only step that does *not* read from `metadata.csv`; it reads from the directory layout produced by Step 8.

### 3.1 Stages at a glance

| Stage | Steps | Reads | Writes | Side effect on disk | Verdict |
|---|---|---|---|---|---|
| Acquire | 3.1–3.5 | 4 public APIs | `data/raw/<src>/<Species>/` | Grows indefinitely | 🟡 Medium |
| Catalogue | 4 | `data/raw/` | in-memory `metadata` DataFrame | None | 🟢 Low |
| Filter | 5–6 | `metadata` | in-memory `metadata` (filtered) | None | 🟡 Medium |
| Split | 7 | `metadata` | in-memory `metadata` (with `split` column) | None | 🟡 Medium |
| Reshape | 8–9 | `metadata` + `data/raw/` | `data/processed/classification_data{,aug}/` | Grows on first run, no-op on re-run | 🟠 High |
| Branch | 10 | `metadata` | `data/processed/detection_data/` | Grows on first run, no-op on re-run | 🔴 Blocker |

---

## 4. Per-Step Findings

Each step is documented with: **purpose** (what it does), **verdict** (overall assessment), **findings** (severity-tagged issues), and **recommended action** (what to do next).

---

### Step 1 — Setup  `md-1` / `code-2` — 🟢 Low

**Purpose.** Imports, fixed `RANDOM_SEED = 42`, and the filesystem helper `safe()`. `safe()` replaces spaces, slashes, and colons with underscores so species names like `Amanita muscaria` become folder-safe strings.

**Verdict.** *Acceptable for a prototype; the seed is set globally for `random` and `numpy.random`.*

**Findings.**

- 🟢 **No Unicode normalization.** `safe()` treats `Morchella` and `MORCHELLA` as different strings, and silently corrupts species names with apostrophes, em-dashes, or non-Latin characters. Effort: **S**. Fix: `unicodedata.normalize('NFKC', name)` before substitution, and reject names with characters outside `[A-Za-z0-9 -]`.

**Recommended action.** No change required for the current species list. Add normalization only if a non-Latin species name is added later.

---

### Step 2 — Paths and configuration  `md-3` / `code-4` — 🟡 Medium

**Purpose.** Defines `RAW_DIR`, `PROCESSED_DIR`, `CLASSIFICATION_DIR`, `DETECTION_DIR`, the 8-species `TARGET_SPECIES` list, and the `SUBSET_PER_SPECIES = 50` cap.

**Verdict.** *Works, but the species list and cap are buried in a notebook cell. A contributor who wants to add a ninth species has to scroll through the notebook.*

**Findings.**

- 🟡 **Hard-coded species list and cap.** Adding/removing a species requires editing a cell mid-notebook; there is no central config. Effort: **S**. Fix: extract to `config.yaml` and load with `pyyaml`.
- 🟢 **No validation of `TARGET_SPECIES`.** Typos like `'Boletus edulus'` (typo) silently filter out everything. Effort: **S**. Fix: assert each entry has two space-separated words and is title-cased correctly.

**Recommended action.** Promote `TARGET_SPECIES` and `SUBSET_PER_SPECIES` to a small YAML file. The cell then becomes a one-liner `cfg = yaml.safe_load(open('config.yaml'))`.

---

### Step 3 — Download a small sample  `step3-main-md` + 5 sub-cells — 🟠 High

**Purpose.** Fetches up to 30 images per (species, source) from iNaturalist, GBIF, FungiTastic, and DF20 into `data/raw/<source>/<Species>/`. Each source has its own fetcher; the orchestrator (3.5) calls them in a loop.

**Verdict.** *Functional, but the error handling is uniformly silent and DF20 cannot auto-download images.*

**Findings.**

- 🟠 **Silent `except: pass` in every fetcher.** A network timeout, a JSON parse error, or a 4xx/5xx response produces the same outcome as a successful run that returned no results: an empty folder. There is no log, no metric, no per-source count of failures. Effort: **M**. Fix: catch per-exception-type, write a per-source `data/raw/<source>/_failures.log` with `(timestamp, url, exception)`, and print a summary at the end of 3.5.
- 🟠 **No retry logic.** A transient 503 or DNS blip stops the cell mid-loop. Effort: **M**. Fix: add `tenacity` with exponential backoff (3 attempts, 1s/2s/4s).
- 🟡 **`API_DELAY = 0.2s` is polite but slow.** 8 species × 4 sources × 30 ≈ 1 000 calls × 0.2 s = ~200 s of pure sleep. Effort: **S**. Fix: parallelize the per-source loop with `concurrent.futures.ThreadPoolExecutor(max_workers=4)`; the polite delay is enforced per-thread.
- 🟡 **DF20 fetcher is a wishlist stub.** The user must manually download a 6.5 GB tarball and extract specific files. The cell documents this clearly, but it is a manual step in an otherwise-automated pipeline. Effort: **L**. Fix: switch to the HuggingFace `datasets` loader for FungiTastic, and accept that DF20 will remain manual.
- 🟡 **Per-source code duplication.** Each fetcher re-implements URL construction, response parsing, and filename generation. Effort: **M**. Fix: a small `SourceFetcher` base class with `fetch(species, limit) -> list[(url, filename)]`.

**Recommended action.** Wrap each `except: pass` with explicit error logging first; the rest can wait. The single biggest quality-of-life win is the failures log — without it, debugging "why does this source have 0 images" is guesswork.

---

### Step 4 — Scan raw folders and build metadata  `md-5` / `code-6` — 🟡 Medium

**Purpose.** Walks `data/raw/`, builds a pandas DataFrame with columns `[image_path, dataset, raw_species, canonical_species]`. The `columns=COLUMNS` argument guarantees the schema even when no images are found.

**Verdict.** *Robust against the empty-data case; not robust against schema drift downstream.*

**Findings.**

- 🟡 **`normalize_species` only handles two-word Latin binomials.** `'Amanita muscaria var. guessowii'` becomes `'Amanita muscaria'` (correct, by accident) but `'Flammulina velutipes'` with stray whitespace would become `'Flammulina  velutipes'` (broken). Subspecies/varieties are silently truncated. Effort: **S**. Fix: split on whitespace, take the first two words, lowercase the second, and drop everything after.
- 🟡 **No schema validation.** Downstream cells (5, 6, 7, 8, 10, 12) will silently misbehave if a column name changes. Effort: **S**. Fix: a `pandera` schema or a few `assert` statements at the end of Step 4.

**Recommended action.** Add a 5-line schema assertion. The cost is zero and it makes future refactors safe.

---

### Step 5 — Filter to target species and subset  `md-7` / `code-8` — 🟡 Medium

**Purpose.** Restricts `metadata` to the 8 target species and caps each (species, source) pair at `SUBSET_PER_SPECIES = 50`.

**Verdict.** *The cap is applied correctly. The selection mechanism is biased.*

**Findings.**

- 🟡 **`.head(SUBSET_PER_SPECIES)` keeps the first N rows in *original* order, not a random sample.** If a single download run produced visually similar images (same photographer, same lighting, same time of day), the subset inherits that bias. Effort: **S**. Fix: `metadata.groupby(['canonical_species', 'dataset'], group_keys=False).sample(n=SUBSET_PER_SPECIES, random_state=RANDOM_SEED)`.

**Recommended action.** One-line change. Until this is fixed, the train/val/test split is reproducible but not representative.

---

### Step 6 — Validate images and deduplicate  `md-9` / `code-10` — 🟡 Medium

**Purpose.** Filters out images smaller than 128 px and removes byte-identical SHA-256 duplicates.

**Verdict.** *The size filter is sensible. The dedup is incomplete.*

**Findings.**

- 🟢 **128 px threshold is appropriate** for MobileNetV4's 224×224 input — small images would be upscaled and lose detail.
- 🟡 **SHA-256 catches byte-identical duplicates only.** A JPEG re-saved at quality 70 vs quality 95 of the same image has a different SHA-256 but is essentially the same image. A cropped/reframed version is also a different file. Effort: **M**. Fix: add a perceptual-hash dedup pass with `imagehash.phash` (Hamming distance ≤ 5).
- 🟡 **No content validation.** A mislabeled or off-topic image (e.g. someone uploaded a picture of their hand) will pass the size and dedup checks. Effort: **L**. Fix: a CLIP zero-shot check against the species name, with a confidence threshold for flagging.

**Recommended action.** The perceptual-hash dedup is the highest-value addition. CLIP content validation can wait.

---

### Step 7 — Train / val / test split  `md-11` / `code-12` — 🟡 Medium

**Purpose.** Stratified 70/15/15 split per species with `random_state=RANDOM_SEED`.

**Verdict.** *Standard and reproducible. With ~30 images per (species, source) pair, the val/test set is too small for stable accuracy estimates.*

**Findings.**

- 🟡 **Single split, no k-fold.** Per-class val/test is ~5–7 images. A 1-image misclassification changes accuracy by 14% on that class. Effort: **M**. Fix: k-fold (k=5) cross-validation on the train+val set, hold out a separate "gold" test set that is touched only at the end.
- 🟡 **Test set is not strictly held out.** If hyperparameters are re-tuned based on test accuracy, the test set leaks. Effort: **S**. Fix: document the rule "test is touched once, at the very end" in the cell markdown, and add a `metadata.test_seen_at` timestamp column to enforce it.

**Recommended action.** Document the holdout rule and add a timestamp. K-fold is a follow-up.

---

### Step 8 — Build the classification dataset  `md-13` / `code-14` — 🟡 Medium

**Purpose.** Copies every image into `data/processed/classification_data/{split}/{Species}/` in the layout `torchvision.datasets.ImageFolder` expects.

**Verdict.** *Simple, fast (~10 000 files/s on local SSD), and idempotent. Doubles disk usage on first run.*

**Findings.**

- 🟡 **No integrity check after the copy.** A copy truncated by a full disk or an interrupted process would silently produce a corrupt training set. Effort: **S**. Fix: compare SHA-256 of source vs dest at the end (already computed in Step 6, can be cached).
- 🟢 **`shutil.copy2` copies metadata; `shutil.copyfile` would be slightly faster** (no-op for this project size).

**Recommended action.** Add the SHA-256 integrity check. Disk-usage growth is acceptable for a prototype.

---

### Step 9 — Resize and augment training data (Albumentations)  `step9-md` / `step9-code` — 🟠 High

**Purpose.** Resizes every image to 224×224 and applies an Albumentations `Compose` pipeline to the training set only. Val/test gets a deterministic `Resize(224, 224)`. Outputs go to a **parallel** `classification_data_aug/` directory, leaving the raw copies in `classification_data/` untouched.

**Pipeline.**

```python
train_pipeline = A.Compose([
    A.RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0, p=0.5),
    A.GaussianBlur(blur_limit=(3, 3), p=0.2),
])
val_pipeline = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE)])
```

This is the right pipeline for mushrooms. The 15° rotation (down from the previous 30°), the `BORDER_REFLECT_101` border (down from white fill), and the `hue=0.0` (no hue jitter, since color is diagnostic) are all deliberate, mushroom-aware choices that the cell's markdown documents well.

**Verdict.** *The pipeline is correct. The implementation has six footguns, none catastrophic but all worth fixing.*

**Findings.**

- 🟠 **`np.random.seed()` is a global side effect.** Albumentations 1.4 removed the per-call `seed` kwarg from `Compose.__call__`, so the cell seeds NumPy's global RNG before each call. This works for a single-threaded dataset-prep step, but it is **not multiprocessing-safe** — if this step is ever parallelized, every worker will see the same seed and produce identical augmentations. Effort: **M**. Fix: use `A.ReplayCompose`, or use Albumentations' `bbox_params` machinery with a per-call seed that is propagated through the transforms themselves.
- 🟠 **`except Exception as e: print(...)` swallows the traceback.** A future failure mode (like the `seed=` kwarg deprecation in Albumentations 1.4) produces a one-line `print` and the loop moves on. Effort: **S**. Fix: `import traceback; traceback.print_exc()` in the except block.
- 🟡 **Idempotency check does not verify the file is a valid JPEG.** `if out_path.exists(): continue` would skip a partial write from a killed run. Effort: **S**. Fix: `Image.open(p).verify()` before skipping (or just `os.remove(p)` and re-write).
- 🟡 **`"_aug" in p.stem` is a fragile filter.** A real source file containing `_aug` (e.g. `inaug_0_0.jpg` from iNaturalist) would be silently skipped. Effort: **S**. Fix: `Path(out_path).name.startswith(stem + "_aug")` so only generated augmented files are skipped.
- 🟡 **`%pip install albumentations` runs on every cell execution.** Pip is idempotent, so the install is a no-op after the first time, but each call costs ~1–2 s of pip startup. Effort: **S**. Fix: move the `%pip install` to a separate "setup" cell at the top of the notebook.
- 🟢 **`A.Resize(224, 224)` distorts aspect ratio for non-square inputs.** Consistent with the previous PIL behavior; not a regression, but worth flagging if the model architecture is later swapped for one that preserves aspect ratio (e.g. a ViT with positional embeddings).

**Recommended action.** Fix the traceback and the `_aug` filter first; the `np.random.seed()` global is acceptable as long as the step is never parallelized. The idempotency `verify()` and the `%pip install` move are low-priority cleanups.

---

### Step 10 — Build the detection dataset (YOLO format)  `step9-det-md` / `step9-det-code` — 🔴 Blocker

**Purpose.** Creates a YOLO-style detection dataset with a single class `mushroom` (id=0). For each image, a label file contains `0 0.5 0.5 1.0 1.0` — a **full-frame placeholder bounding box**.

**Verdict.** *The dataset is structurally correct. The annotations are degenerate.*

**Findings.**

- 🔴 **Placeholder full-frame bboxes teach the detector to ignore localization.** A model trained on `0 0.5 0.5 1.0 1.0` for every image will converge to predicting a full-frame box regardless of input. The cell's markdown acknowledges this ("crude annotations; replace them with real bounding boxes later for better results"), but the *current state* is a degenerate baseline, not a useful detector. Effort: **L** (requires labeling UI, human review, or a pre-trained mushroom detector to bootstrap). Fix: replace with real bboxes from CVAT, LabelImg, or Roboflow.
- 🟠 **The detection dataset is not augmented.** Step 9 augments the classification set but not the detection set (because augmenting requires valid bboxes that transform with the image). The detector therefore sees no rotation/scale variation in training and relies on YOLO11n's built-in `RandomAffine`. This is fine *as a baseline* but means the detector is at the mercy of the built-in defaults.
- 🟠 **All 8 species are collapsed into a single `mushroom` class.** The detector cannot distinguish species. If the downstream task is species-level localization (e.g. "is this Amanita muscaria in the frame?"), the dataset is structurally insufficient.

**Recommended action.** Replace placeholder bboxes with real ones before any detector training run. Even 50–100 hand-labeled images per species would be a meaningful improvement.

---

### Step 11 — Explore the data  `step-8-md` / `step-8-code` — 🟡 Medium

**Purpose.** Two visualizations: a per-species-per-dataset count matrix, and a 3-column image grid (original / preprocessed / augmented).

**Verdict.** *Useful for debugging. The augmented column embeds the OLD pipeline, which is now misleading.*

**Findings.**

- 🟡 **The augmented column shows the OLD PIL white-fill pipeline**, not the current Albumentations pipeline. The visualization is now stale. Effort: **M**. Fix: regenerate from a versioned `scripts/viz_explore.py` that uses the Step 9 pipeline.
- 🟡 **The visualization is an embedded base64 image** in the notebook output, so it is not diffable in git. Effort: **M**. Fix: write the figure to `docs/figures/explore.png` and embed a markdown link.

**Recommended action.** Move the visualization to a versioned script and a saved figure. The current state is a one-off that drifts from the code.

---

### Step 12 — Save metadata  `md-15` / `code-16` — 🟠 High

**Purpose.** Writes the per-image `metadata` DataFrame to `data/processed/metadata.csv` for downstream notebooks to consume.

**Verdict.** *Minimal. The CSV has no schema, no version, and is overwritten in place.*

**Findings.**

- 🟠 **No `pipeline_version` column or sidecar version file.** A downstream notebook has no way to know which version of the preprocessing pipeline produced the CSV. If a column is added in Step 4 six months from now, `train_classification.ipynb` will silently misread the old CSV. Effort: **S**. Fix: add a `pipeline_version` column with a constant string, or write a sidecar `metadata.version` file with a semver.
- 🟠 **No backup of the previous CSV before overwriting.** A re-run with different filters silently destroys the old metadata. Effort: **S**. Fix: write to `metadata_<timestamp>.csv` first, then atomically promote to `metadata.csv` after a successful run.
- 🟡 **Path and column types are not preserved.** `image_path` is a string; on Windows, the path separator changes on round-trip. `dataset` could be misread as int if all values are numeric in a future dataset. Effort: **S**. Fix: use Parquet instead of CSV, or document the type contract in the cell markdown.

**Recommended action.** Add the version column and the atomic-promote pattern. Parquet is a follow-up.

---

### Step 13 — Next steps  `md-17` (markdown only) — 🟠 High

**Purpose.** Three bullet-point handoff instructions to the training notebooks.

**Verdict.** *Stale.*

**Findings.**

- 🟠 **The "open `train_classification.ipynb` and point it at `processed/classification_data/`" instruction is wrong.** The augmented dataset lives at `processed/classification_data_aug/`, not `processed/classification_data/`. Following the cell's guidance literally would load the wrong (non-augmented) images. Effort: **S**. Fix: update the path.
- 🟠 **"When you have segmentation masks, build the detection dataset" is now redundant.** Step 10 already creates the detection dataset (with placeholder bboxes). Effort: **S**. Fix: remove the bullet and replace with "Step 10 has placeholder bboxes; replace them before detector training."

**Recommended action.** Update the markdown. Both fixes are one-line edits.

---

## 5. Cross-Cutting Findings

These are issues that span multiple steps and would not be visible from a per-step audit alone.

### 5.1 Two competing notions of "done"

The pipeline has two output trees (`classification_data/` and `classification_data_aug/`) that are *almost* the same except for the augmentations on `train`. There is no machine-readable way to know which one the training notebook should load — the convention is encoded in prose in Steps 12 and 13. A simple `manifest.json` in `data/processed/` listing the contents and intended use of each subdirectory would eliminate the ambiguity.

### 5.2 Determinism is partial

The pipeline is deterministic **up to Step 8** (every cell uses `RANDOM_SEED` and `out_path.exists()` idempotency). From Step 9 onward, determinism is global-state-dependent (`np.random.seed()`). If a contributor reorders Step 9 and Step 10, the *names* of the augmented files would change (because `_aug{k}` is per-iteration), and the *content* would be different (because the global RNG state differs). This is a hidden coupling.

### 5.3 No metrics on the pipeline itself

There is no way to tell, from the outputs alone, how long Step 3.5 took, how many requests failed, how many duplicates were removed in Step 6, or how many augmented images were written in Step 9. A small `data/processed/_pipeline_stats.json` with `{step_name: {n_input, n_output, n_failed, duration_s}}` would be a meaningful debugging aid and would cost ~5 lines per step.

### 5.4 Idempotency relies on string matching

Three steps (8, 9, 10) use `if dst.exists(): continue` to skip already-written files. None of them verify that the existing file is valid. A contributor who manually deletes a single file from `data/processed/classification_data_aug/` will get a re-run that silently re-creates only the missing one — fine. A contributor whose disk filled up mid-run and got a truncated file will get a re-run that *skips* the truncated file — bad.

### 5.5 The metadata CSV is consumed by the future, not the present

`metadata.csv` is written in Step 12 but is not read by any subsequent cell in `data_preprocessing.ipynb`. The training notebooks are the only consumers, and they read images by glob, not by the CSV. This is fine for the current pipeline but makes the CSV a "documentation artifact" rather than a "data contract." Either remove the CSV (and put the metadata in a `manifest.json` that the pipeline itself reads on re-run), or commit to the CSV as a real data contract (with schema, version, and integrity checks).

---

## 6. Recommendations Roadmap

Prioritized by impact-per-effort. Each item is anchored to the cell ids it would touch.

### P0 — Before the next training run

| # | Item | Touches | Effort |
|---|---|---|---|
| 1 | Replace Step 10 placeholder bboxes with real annotations (or accept the dataset as a baseline-only artifact) | `step9-det-code` | L |
| 2 | Fix Step 13 handoff path: `classification_data` → `classification_data_aug` | `md-17` | S |
| 3 | Add `pipeline_version` column to `metadata.csv` and use atomic-promote (write to `metadata_<ts>.csv`, then rename) | `code-16` | S |
| 4 | Add `traceback.print_exc()` to the Step 9 except block | `step9-code` | S |
| 5 | Switch Step 5 subset selection from `.head()` to `.sample(random_state=RANDOM_SEED)` | `code-8` | S |

### P1 — Next iteration

| # | Item | Touches | Effort |
|---|---|---|---|
| 6 | Per-source failure logging in Step 3 (one `data/raw/<src>/_failures.log` per source) | `step3-1..5-code` | M |
| 7 | `tenacity` retry with exponential backoff in Step 3 | `step3-1..5-code` | M |
| 8 | Switch the Step 9 `_aug` filter to `Path(out_path).name.startswith(stem + "_aug")` | `step9-code` | S |
| 9 | Add SHA-256 integrity check at the end of Step 8 | `code-14` | S |
| 10 | Add `pandera` schema (or `assert` statements) at the end of Step 4 | `code-6` | S |
| 11 | Perceptual-hash dedup (imagehash.phash, Hamming ≤ 5) in Step 6 | `code-10` | M |
| 12 | Document the "test set touched once" rule in Step 7 markdown | `md-11` | S |

### P2 — Backlog

| # | Item | Touches | Effort |
|---|---|---|---|
| 13 | `A.ReplayCompose` for per-call determinism in Step 9 | `step9-code` | M |
| 14 | Move `%pip install albumentations` to a separate setup cell | `step9-code` | S |
| 15 | Aspect-ratio-preserving val/test resize (`SmallestMaxSize` + `CenterCrop`) | `step9-code` | S |
| 16 | K-fold cross-validation in Step 7 | `code-12` | M |
| 17 | Re-generate Step 11 visualization from a versioned `scripts/viz_explore.py` | `step-8-code` | M |
| 18 | Promote `TARGET_SPECIES` and `SUBSET_PER_SPECIES` to `config.yaml` | `code-4` | S |
| 19 | Emit `data/processed/_pipeline_stats.json` with per-step counts and durations | `code-6, 8, 10, 12, 14, step9-code` | M |
| 20 | Refactor Step 3 fetchers to a small `SourceFetcher` base class | `step3-1..5-code` | M |

---

## 7. What Works Well

A balanced audit acknowledges strengths. The pipeline has several.

- **The cell-id convention is consistent.** Most cells use `md-N` / `code-N`; the multi-cell Step 3 uses `step3-N-md` / `step3-N-code`; Steps 9–10 use `step9-` and `step9-det-` prefixes. This makes it possible to find a cell by its id alone, which is exactly what this report relies on.
- **Idempotency is implemented correctly at the directory level.** Every step that writes to disk checks for existing files first. A re-run after a successful run is a no-op (verified during the audit by re-running Step 9 against a populated `classification_data_aug/`).
- **The classification augmentation pipeline is mushroom-aware.** 15° rotation, no hue jitter, `BORDER_REFLECT_101` border, augment-then-resize — every choice is defensible against the 2024-2025 best-practice guidance for fine-grained mushroom classification. The cell's markdown documents the reasoning, which is rare and valuable.
- **The classification and detection datasets share raw images but write to parallel trees.** A buggy augmentation step cannot corrupt the detection dataset, and a buggy detection step cannot corrupt the classification dataset. This separation is exactly the right shape for a project that wants to evolve one path without risk to the other.
- **The notebook is end-to-end runnable.** With the data sources in place, the notebook produces the two `data/processed/` trees and the `metadata.csv` in a single execution. There is no "and then run this other script" hidden in the markdown.
- **The audit's cross-references (cell ids, file paths, code spans) are exact.** A contributor can `grep` for any cell id in this report and jump to the relevant code in the notebook. The cell ids are the API between the report and the code.

---

## 8. Appendix A — Source References

The audit draws on the following primary sources. Each is current as of 2024–2025 and reflects the state of the art at the time of writing.

- **Albumentations 1.4 documentation** ([albumentations.ai/docs](https://albumentations.ai/docs/)) — `Compose`, `BboxParams`, `border_mode` (used for `BORDER_REFLECT_101`), and the v1.4 removal of the per-call `seed` kwarg.
- **torchvision transforms v2 documentation** ([docs.pytorch.org/vision/stable/transforms.html](https://docs.pytorch.org/vision/stable/transforms.html)) — alternative transform API; not currently used, but noted as a future migration option for the detection pipeline.
- **fastai vision augmentation** ([docs.fast.ai/vision.augment.html](https://docs.fast.ai/vision.augment.html)) — the "augment-then-resize" pattern (presizing), which Step 9 implements.
- **ultralytics YOLO11 documentation** ([docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)) — the `yolo.yaml` schema and the `RandomAffine` defaults that the detection dataset relies on.
- **Kumar et al. 2025**, *Enhancing Image Classification with Augmentation* ([arXiv:2502.18691](https://arxiv.org/abs/2502.18691)) — fine-grained classification survey, in particular the section on hue preservation for color-diagnostic classes.
- **Pillow (PIL) documentation** ([pillow.readthedocs.io](https://pillow.readthedocs.io/)) — `Image.LANCZOS` (used in the previous PIL pipeline), `Image.BILINEAR` (used by the original `_augment`), and `Image.open(p).verify()` (recommended for the Step 9 idempotency fix).

## 9. Appendix B — Glossary

| Term | Definition |
|---|---|
| **Bbox** | Bounding box. In YOLO format: `class_id x_center y_center width height`, all normalized to `[0, 1]`. |
| **BORDER_REFLECT_101** | An OpenCV border mode that reflects pixel values at the image edge (`gfedcb|abcdefgh|gfedcba`); used to avoid the white-fill artifact of constant-pad rotations. |
| **`colorjitter`** | A geometric-preserving color transform that randomly perturbs brightness, contrast, saturation, and hue. For mushrooms, **hue is set to 0.0** because a small hue shift can change a red `Amanita muscaria` to orange or a yellow `Cantharellus cibarius` to green — exactly the signal we want to preserve. |
| **Determinism** | A pipeline is deterministic if re-running it with the same inputs and the same seed produces byte-identical outputs. The current pipeline is deterministic up to Step 8; from Step 9 onward it is global-state-dependent. |
| **HuggingFace `datasets`** | A library for streaming and downloading ML datasets; the recommended way to consume FungiTastic without manually downloading a 150 GB tarball. |
| **Idempotency** | A step is idempotent if re-running it on already-populated outputs is a no-op. The current pipeline is idempotent at the directory level (`if out_path.exists(): continue`) but not at the file-validity level (no `Image.verify()` check). |
| **`np.random.seed()`** | Seeds NumPy's *global* random state. The reason Step 9 uses it is that Albumentations 1.4 removed the per-call `seed` kwarg from `Compose.__call__`. The global side effect is fine for a single-threaded dataset-prep step, but it is not multiprocessing-safe. |
| **PANdas / pandera** | pandas is the DataFrame library. pandera is a runtime DataFrame schema validator; using it (or a few `assert` statements) at the end of Step 4 would catch schema drift downstream. |
| **Perceptual hash (phash)** | A fingerprint of an image's content that is robust to small changes (compression, resize, crop). Used in Step 6's recommended P1 fix to catch near-duplicates that SHA-256 misses. |
| **ReplayCompose** | An Albumentations feature that records the parameters of a stochastic pipeline so the *same* augmentation can be reproduced on a different image. Useful for per-call determinism in Step 9's P2 fix. |
| **RGB** | Red, Green, Blue. The image is loaded as `Image.open(path).convert('RGB')` to ensure 3-channel uint8 input, which is what Albumentations expects. |
| **Stratified split** | A train/val/test split that preserves the per-class ratio. Implemented in Step 7 via `train_test_split(..., stratify=metadata['canonical_species'])`. |
| **`tenacity`** | A Python library for retrying with exponential backoff. The recommended P1 fix for Step 3's silent failure mode. |
| **YOLO** | You Only Look Once. A single-stage object detector. The YOLO dataset convention has parallel `images/` and `labels/` directories with matching filenames. |
