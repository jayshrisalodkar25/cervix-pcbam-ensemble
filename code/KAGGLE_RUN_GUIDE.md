# How to run the revision experiments on Kaggle

## 1. One-time setup (2 minutes)
1. Kaggle → **New Notebook**, then paste in the cells of `cervix_revision.py` (the `# %% [n]`
   markers show the original cell boundaries).
2. **Add Input** → the three datasets:
   - `bhavanamohod16/lbc-dataset1` (Mendeley LBC)
   - `bhavanamohod16/malhari` (Malhari)
   - `prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed` (SIPaKMeD)
   The notebook finds them automatically wherever Kaggle mounts them.
3. Settings → **Accelerator: GPU T4 x2 or P100**, **Internet: On** (needed to download the ImageNet weights).
4. **Save Version → Save & Run All (Commit)**. Do not use interactive mode for the long run: the session would time out.

## 2. Sessions and runtime (estimate for one T4/P100)
| Session | What finishes | Approx. GPU time |
|---|---|---|
| 1 | Audit, splits, SIPaKMeD pretraining, seed 0 of LBC (ImageNet + SIPaKMeD), CORAL/MK-MMD, few-shot; start of multi-domain | ~11 h |
| 2 | Multi-domain (ImageNet + SIPaKMeD), two-stage cascade, seed 1 | ~11 h |
| 3 | Seed 2, baselines (ResNet-50, DenseNet-121, EfficientNet-B3 without CBAM) | ~8–11 h |

The notebook stops starting new trainings after 11 h and always runs the report cells, so each session gives
usable (partial) tables. **To continue:** open the notebook → Add Input → *Notebook Output* → select this
notebook's latest version → Save & Run All again. Finished models are skipped automatically.

Kaggle gives ~30 GPU hours per week, so plan for 1–2 weeks. If you are short of time, the most important
results come from session 1, which covers editor points 1–3.

## 3. What the notebook GENERATES (no manual work)
Everything is in `/kaggle/working/revision_outputs/`. Download `manuscript_bundle.zip`, then unzip it into the
manuscript folder (it creates `generated/` and `figures/`) and recompile `main.tex`.

| Editor request | File generated | Goes into the manuscript as |
|---|---|---|
| 1 Splitting strategy, ratios, leakage check | `generated/tab_splits.tex`, `splits/*.csv`, `audit/audit_summary.json`, `audit/duplicate_pairs.csv`, `audit/duplicate_pairs_montage.png` | Table 1 + Sec. 3.3 numbers |
| 1 SIPaKMeD on held-out data (replaces the 100% table) | `generated/tab_sipakmed.tex` | Table 5 |
| 1/3 All main results, multi-seed mean ± SD | `generated/tab_main.tex`, `results/table_main.csv` | Table 7 |
| 3 Seeds, CIs, statistical tests (CORAL/MK-MMD, few-shot, ImageNet vs SIPaKMeD, entropy vs uniform) | `generated/tab_da.tex`, `results/table_statistics.csv` | Table 8 |
| 2 ECE, Brier, NLL, AURC, error-AUROC, uniform vs entropy-weighted, τ sweep | `generated/tab_calibration.tex`, `results/table_calibration.csv` | Table 9 |
| 2 Reliability diagrams | `figures/*_reliability.pdf` | Figure 8 |
| 4 Confusion matrices (LBC, Malhari, CORAL/MK-MMD, few-shot, multi-domain, cascade) | `figures/*_cm.pdf` | Figure 9 |
| 4 Same-protocol baselines + attention ablation | `generated/tab_baselines.tex` | Table 10 |
| 4 Training/inference time and memory, 1 vs 5 models | `generated/tab_cost.tex` | Table 13 |
| 4/5 Two-stage cascade vs flat, SCC recall with CI | `generated/tab_cascade.tex`, `figures/cascade_lbc_cm.pdf` | Table 6 |
| Figure 4 replacement (train + validation curves) | `figures/learning_curves.pdf` | Figure 4 (picked up automatically) |
| t-SNE before/after CORAL/MK-MMD | `figures/tsne_da.pdf` | Figure 5 (picked up automatically) |
| Numbers for the orange [placeholders] in the text | `results/VALUES_FOR_MANUSCRIPT.txt`, `results/values_for_manuscript.json` | Abstract, Sec. 3.3, 4.1, 4.2, 5, Conclusion, response letter |

## 4. What the notebook does NOT generate (you must do it)
1. **Type the text numbers.** Copy each value from `VALUES_FOR_MANUSCRIPT.txt` into the orange `\tbd{...}`
   places in `main.tex` and `response_to_reviewers.tex`. Search for `\tbd` to find every one of them.
2. **Rewrite sentences that depend on the outcome.** Example: if entropy weighting is not better than uniform
   averaging, say so (the text explains how).
3. **Check the duplicate montage by eye** (`audit/duplicate_pairs_montage.png`) and state the result in Sec. 3.3.
4. **Redraw Figure 2.** SIPaKMeD needs its own 5 cell-type labels, the "binnary" typo must be fixed, and the
   drawing should show CBAM at the 48/136/384/1536-channel positions plus the projection head.
5. **Malhari metadata** (preparation type, magnification, microscope) in Sec. 3.7.2.
6. **Literature table "Split reported" column.** Check each cited paper and write image / patient / not stated.
7. **Full Reviewer 1 and 2 comments.** Paste them into the response letter and answer each one.
8. **Patient IDs.** These cannot be generated. Optionally email the Mendeley LBC authors; if you receive IDs, put
   them in the `group` column of `splits` and re-run.
9. **Figures 6–7 (old t-SNE).** Keep them or delete them; they come from the old protocol. Figure 5 is regenerated.
10. **Code/data URL, author contributions, and author-name consistency** (Shelke vs Salodkar).
11. **Move the text into the journal's sn-jnl template** and upload the marked + clean PDFs.

## 5. Settings you may change (optional)
They are at the top of cell [1] (`CFG`):
- `SEEDS_MAIN` (default `[0, 1, 2]`): use `[0, 1, 2, 3, 4]` for five seeds if you have the GPU time.
- `RUN_BASELINES` / `RUN_CASCADE`: set to `False` to save time. If the cascade is not run, remove the two-stage
  claim from the paper.
- `MAX_HOURS`: lower it if you need the session to end earlier.

## 6. What is identical to your original code, and what is deliberately different
- **Identical:** the DeepCervixModel architecture (11.94 M parameters, the same value as your Table 7), the
  focal loss, AdamW + OneCycleLR, learning rate 3e-4, batch 32, 300 px, augmentation, and early stopping with
  patience 7.
- **Different, to satisfy the reviewers:**
  - group-aware splits with an untouched 20% test set
  - SIPaKMeD weights are verified on load, and the run stops if they are missing
  - few-shot early stopping uses a validation subset, not the test set
  - CORAL/MK-MMD runs for a fixed 10 epochs on the 256-d embeddings, with a median-heuristic MMD
    (the old fixed-σ MMD gave a constant 0.125 loss)
  - class weights come from training labels only
  - real five-fold ensembles (uniform and entropy-weighted) with temperature scaling
  - 3 seeds and paired statistical tests
