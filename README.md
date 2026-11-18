# Uncertainty-Aware Homogeneous Ensemble Learning with Progressive Attention for Cervical Cytology Classification under Cross-Dataset Domain Shift

Code and supplementary results for the above manuscript.

Authors: Jayshri Shelke, A. A. Khurshid
Department of Electronics Engineering, Ramdeobaba University, Nagpur, India

---

## What is in this repository

```
code/                        analysis code (see "How the code is organised")
  cervix_revision.py         main pipeline: audit, splits, training, evaluation, tables, figures
  reviewer2_addon.py         additional analyses (aggregation ablation, risk-coverage, per-class CIs)
  sipakmed_check.py          SIPaKMeD data checks and cluster-grouped classifier
  sip2_retrain.py            re-training of the SIPaKMeD-initialised models
  round2_additions.py        majority-class baseline and exact Stage-1 sensitivity CI
  KAGGLE_RUN_GUIDE.md        how the experiments were run end to end

supplementary/               outputs referenced by the manuscript
  splits/                    image lists for every partition and fold (lbc, malhari target, merged, sipakmed)
  audit/                     duplicate audit: flagged pairs, excluded images, groups, summary, montage
  results/                   numerical results behind the tables (CSV/JSON) + VALUES_FOR_MANUSCRIPT.txt
  generated/                 LaTeX tables generated from those results
  figures/                   confusion matrices, reliability diagrams, t-SNE, learning curves, ROC/PR
  review2/                   extended aggregation ablation, uncertainty and failure analyses
  sipakmed_check/            SIPaKMeD data checks and cluster-grouped results
  sip2/                      SIPaKMeD-initialised re-training results
```

## How the code is organised

The experiments were developed and run as a **Kaggle notebook**. The files in
`code/` are that notebook exported as scripts, with the original cell numbers
preserved in the `# %% [n]` markers. This matters for how they are run:

| File | Cells | Runs |
|---|---|---|
| `cervix_revision.py` | `[1]`–`[16]` | first; self-contained |
| `reviewer2_addon.py` | `[17]` | after `[1]`–`[16]`, in the same session |
| `sipakmed_check.py` | `[18]` | after `[1]`–`[16]`, in the same session |
| `sip2_retrain.py` | `[19]` | after `[18]` |
| `round2_additions.py` | — | standalone, any time |

**`reviewer2_addon.py`, `sipakmed_check.py` and `sip2_retrain.py` are
continuation cells, not standalone programs.** They reuse objects defined in
`cervix_revision.py` (the output directory `OUT`, the `log()` helper, the
saved per-seed predictions loaded by `load_seed()`, and the temperature and
aggregation helpers). Running them on their own raises `NameError`. Paste them
as further cells of the same notebook, or run them in a session that has
already executed `cervix_revision.py`.

`round2_additions.py` is the exception: it is a plain script with no
dependency on the rest, because it only re-derives two closed-form quantities
from class counts already reported in the manuscript. Run it with
`python3 round2_additions.py` to reproduce:

- the majority-class baseline on the Malhari target-test set
  (52.7% accuracy, 25.0% balanced accuracy), and
- the exact Clopper–Pearson 95% confidence interval for the Stage-1
  sensitivity of 70/70 abnormal images (94.9%–100.0%).

## Reproducing the experiments

See `code/KAGGLE_RUN_GUIDE.md` for the full procedure (accelerator settings,
session budgeting and resuming). In outline:

1. Create a Kaggle notebook and paste in the cells of `cervix_revision.py`.
2. Attach the three public datasets listed below as notebook inputs; the code
   locates them automatically wherever Kaggle mounts them.
3. Set the accelerator to GPU (T4/P100) and Internet to On (for ImageNet
   weights), then Save & Run All.
4. Outputs are written to `/kaggle/working/revision_outputs/` and bundled into
   zip files; `supplementary/` in this repository is the contents of those
   bundles for the run reported in the manuscript.

The full experiment matrix takes roughly 30 GPU-hours; the code is resumable
and skips completed trainings, so it can be run across several sessions.

### Dependencies

```
torch torchvision timm albumentations scikit-learn scipy pandas matplotlib pillow
```
`round2_additions.py` needs only `scipy`.

## Data

The three datasets are public and are **not** redistributed here. They are
cited in the manuscript and were obtained from:

| Dataset | Source |
|---|---|
| Mendeley LBC (Hussain et al.) | https://doi.org/10.17632/zddtpgzv63 |
| Malhari (Kalbhor) | https://doi.org/10.17632/m5kxdj7m36.1 |
| SIPaKMeD (Plissiti et al.) | original publication; Kaggle mirror used in this work |

The Kaggle mirrors used for the copies in this work are listed in the Data
availability statement of the manuscript. The repository records do not
specify a licence beyond public availability for research; the datasets were
used for non-commercial research with citation of the original publications,
as requested by the providers.

`supplementary/splits/` gives the exact partition and fold assignment of every
image used, and `supplementary/audit/` gives the duplicate audit, so the
splits reported in the manuscript can be reconstructed exactly from the public
data without re-running the audit.

## Notes on reported results

- Temperature `τ = 0.5` was fixed in advance; the τ sweep in the results is a
  sensitivity analysis and was never used for model selection.
- The held-out test sets are read only by the evaluation code; nothing in
  training reads them.
- The Mendeley LBC, CORAL/MK-MMD and few-shot models use three random seeds;
  the multi-domain models, the two-stage cascade and the same-protocol
  baselines use a single seed, for the computational reasons stated in the
  manuscript. Comparisons involving the single-seed baselines are therefore
  reported descriptively and are not significance-tested.
