"""
Second-round revision additions (no retraining, no raw data needed).

These are the only two numbers introduced in the second revision that were not
already produced by cervix_revision.py / reviewer2_addon.py in the first
round. Both are closed-form computations from numbers already reported in the
manuscript's own tables, kept here as a standalone, auditable script so a
reviewer or reader can re-derive them in seconds.

Run:  python3 round2_additions.py
"""
from scipy import stats


def majority_class_baseline(class_counts):
    """Reviewer Point 1: majority-class baseline on the Malhari target-test set.

    class_counts: dict of {class_name: n_images}, from Table 14 of the
    manuscript (Malhari target-test, Mendeley-LBC-only training row):
        NILM=116, LSIL=59, HSIL=30, SCC=15  (total 220)
    """
    total = sum(class_counts.values())
    majority_n = max(class_counts.values())
    accuracy = majority_n / total
    # Balanced accuracy of an "always predict the majority class" rule:
    # the majority class gets 100% recall, every other class gets 0% recall,
    # so balanced accuracy = 1 / (number of classes).
    balanced_accuracy = 1.0 / len(class_counts)
    return accuracy, balanced_accuracy


def stage1_sensitivity_ci(x, n, alpha=0.05):
    """Reviewer Point 5: exact (Clopper-Pearson) 95% CI for Stage-1 sensitivity.

    x: number of abnormal images correctly screened as abnormal by Stage 1
    n: total number of abnormal images in the held-out Mendeley LBC test set
       (Table 14: LSIL 25 + HSIL 30 + SCC 15 = 70)
    An exact binomial interval is used, rather than a bootstrap percentile
    interval, because x = n = 70 (100% observed sensitivity) is a boundary
    case where percentile bootstrap CIs collapse to a point; Clopper-Pearson
    remains well defined and is the standard choice for this situation.
    """
    lo = stats.beta.ppf(alpha / 2, x, n - x + 1) if x > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, x + 1, n - x) if x < n else 1.0
    return lo, hi


if __name__ == "__main__":
    counts = {"NILM": 116, "LSIL": 59, "HSIL": 30, "SCC": 15}
    acc, bal_acc = majority_class_baseline(counts)
    print("Majority-class baseline (Malhari target-test, n=%d):" % sum(counts.values()))
    print(f"  accuracy          = {acc * 100:.1f}%")
    print(f"  balanced accuracy = {bal_acc * 100:.1f}%")
    print()

    x, n = 70, 70  # Mendeley LBC held-out test set: 25 LSIL + 30 HSIL + 15 SCC
    lo, hi = stage1_sensitivity_ci(x, n)
    print(f"Stage-1 sensitivity: {x}/{n} = {x / n * 100:.2f}%")
    print(f"Exact (Clopper-Pearson) 95% CI: [{lo * 100:.1f}%, {hi * 100:.1f}%]")
