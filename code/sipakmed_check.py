# %% [18] SIPaKMeD check and properly trained cluster-grouped classifier (Reviewer 2, comment 7)
# Run after cells [1]-[16] (cell [17] is not needed). Outputs: OUT/sipakmed_check/ and OUT/sipakmed_check.zip
#   A  data checks: class counts per partition, cluster leakage, identical files across partitions, image sizes,
#      learning curve of the original 15-epoch pretraining run, example cells per class
#   B  cluster-grouped classifier trained with the main training recipe (OneCycle, up to 40 epochs, focal loss,
#      early stopping not before epoch 15), evaluated once on the held-out cells (with flip test-time augmentation)
#   C  reference: the same recipe on a random cell-level split (cells of one cluster image may fall in train and
#      test) -> quantifies how much the leaky split inflates the result
# Time on a T4/P100: A ~2 min, B ~20-30 min, C ~20-30 min.
import hashlib, traceback
from sklearn.model_selection import StratifiedShuffleSplit

RUN_LEAKY_REFERENCE = True
SC = f"{OUT}/sipakmed_check"; os.makedirs(SC, exist_ok=True)
SCV = {}
NAMES = [c.replace("im_", "") for c in SIP_CLASSES]


def _sc_section(name, fn):
    t0 = time.time()
    try:
        fn(); log(f"sipakmed_check: {name} done ({time.time() - t0:.0f} s)")
    except Exception:
        log(f"sipakmed_check: {name} FAILED"); traceback.print_exc()


def predict_tta(model, df, classes):
    """mean softmax over the image, its horizontal flip and its vertical flip"""
    model.eval(); ps = []
    with torch.no_grad():
        for x, _ in loader(df, classes, train=False, shuffle=False):
            x = x.to(DEVICE)
            with autocast():
                p = sum(torch.softmax(model(v).float(), 1) for v in (x, torch.flip(x, [3]), torch.flip(x, [2]))) / 3
            ps.append(p.cpu())
    return torch.cat(ps).numpy()


def _report(tag, y, p):
    r = metrics(y, p, SIP_CLASSES)
    ci = boot_ci(y, p, lambda a, b: accuracy_score(a, b.argmax(1)))
    out = {"accuracy": r["accuracy"], "accuracy_ci": ci, "balanced_accuracy": r["balanced_accuracy"],
           "macro_f1": r["f1"], "macro_auc": r["auc"], "n": int(len(y)), "cm": r["cm"],
           "per_class": {n: {"recall": r[f"recall_{c}"], "precision": r[f"precision_{c}"], "support": r[f"support_{c}"]}
                         for n, c in zip(NAMES, SIP_CLASSES)}}
    SCV[tag] = out
    log(f"sipakmed_check {tag}: accuracy {100*out['accuracy']:.2f}% [{100*ci[0]:.1f}, {100*ci[1]:.1f}], "
        f"balanced {100*out['balanced_accuracy']:.2f}%, macro-F1 {out['macro_f1']:.4f}, macro-AUC {out['macro_auc']:.4f}")
    cm = np.array(out["cm"])
    fig, a = plt.subplots(figsize=(6.5, 5.5)); a.imshow(cm, cmap="Blues")
    for i in range(len(NAMES)):
        for j in range(len(NAMES)):
            a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_xticks(range(len(NAMES))); a.set_xticklabels(NAMES, rotation=30, ha="right")
    a.set_yticks(range(len(NAMES))); a.set_yticklabels(NAMES); a.set_xlabel("Predicted"); a.set_ylabel("True")
    a.set_title(f"SIPaKMeD held-out cells ({tag}): accuracy {100*out['accuracy']:.1f}%")
    plt.tight_layout(); plt.savefig(f"{SC}/cm_{tag}.png", dpi=200); plt.savefig(f"{SC}/cm_{tag}.pdf"); plt.close()
    return out


# ------------------------------------------------------------------ A  data checks
def part_a_checks():
    s = SPL_SIP.copy()
    print(pd.crosstab(s.partition, s.label, margins=True))
    check_no_leak(s)                                     # raises if a cluster image spans two partitions
    SCV["n_cells"] = int(len(s)); SCV["n_clusters"] = int(s.group.nunique())
    SCV["clusters_crossing_partitions"] = int((s.groupby("group").partition.nunique() > 1).sum())
    md5 = s.path.map(lambda p: hashlib.md5(open(p, "rb").read()).hexdigest())
    cross = s.assign(md5=md5).groupby("md5").partition.nunique()
    SCV["identical_files_across_partitions"] = int((cross > 1).sum())
    sizes = s.path.map(lambda p: Image.open(p).size)
    s["w"], s["h"] = [x[0] for x in sizes], [x[1] for x in sizes]
    SCV["median_size_by_class"] = s.groupby("label")[["w", "h"]].median().round(0).to_dict()
    print("clusters crossing partitions:", SCV["clusters_crossing_partitions"],
          "| identical files across partitions:", SCV["identical_files_across_partitions"])
    print("median crop size (px):", SCV["median_size_by_class"])
    lg = f"{run_dir('sip_pre', 0, 0)}/log.json"
    if os.path.exists(lg):
        h = json.load(open(lg))["history"]
        SCV["original_run_val_f1_by_epoch"] = [round(e["val_f1"], 4) for e in h]
        print("original 15-epoch pretraining, validation macro-F1 per epoch:", SCV["original_run_val_f1_by_epoch"])
    fig, ax = plt.subplots(len(SIP_CLASSES), 6, figsize=(12, 10))
    te = s[s.partition == "test"]
    for i, c in enumerate(SIP_CLASSES):
        ex = te[te.label == c].sample(min(6, (te.label == c).sum()), random_state=0)
        for j, a in enumerate(ax[i]):
            a.axis("off")
            if j < len(ex):
                a.imshow(Image.open(ex.path.values[j]).convert("RGB"))
        ax[i, 0].set_title(NAMES[i], fontsize=10, loc="left")
    plt.tight_layout(); plt.savefig(f"{SC}/example_test_cells.png", dpi=120); plt.close()


# ------------------------------------------------------------------ B  cluster-grouped classifier, main recipe
def _train_eval(tr, va, te, tag, seed=0):
    warm_cache(pd.concat([tr, va, te]).path)
    seed_all(seed)
    m = build_model("pcbam", len(SIP_CLASSES), "imagenet")
    m, info = fit(m, tr, va, SIP_CLASSES, epochs=40, lr=CFG["LR"], patience=CFG["PATIENCE"], bs=CFG["BS"],
                  loss="focal", schedule="onecycle", min_epochs=15)
    SCV[f"{tag}_epochs"] = info["epochs_run"]; SCV[f"{tag}_best_val_f1"] = info["best_val_f1"]
    SCV[f"{tag}_val_f1_by_epoch"] = [round(e["val_f1"], 4) for e in info["history"]]
    y = np.array([SIP_CLASSES.index(l) for l in te.label])
    out = _report(tag, y, predict_tta(m, te, SIP_CLASSES))
    return m, out


def part_b_grouped():
    s = SPL_SIP
    tr, va, te = s[s.partition == "dev"], s[s.partition == "val"], s[s.partition == "test"]
    m, _ = _train_eval(tr, va, te, "cluster_grouped")
    torch.save(m.state_dict(), f"{SC}/sipakmed_cluster_grouped_v2.pth")
    del m; torch.cuda.empty_cache()


# ------------------------------------------------------------------ C  leaky random cell-level split (reference)
def part_c_leaky():
    s = SPL_SIP.reset_index(drop=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=46)
    dev_i, te_i = next(sss.split(np.zeros(len(s)), s.label))
    dev = s.loc[dev_i].reset_index(drop=True)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=47)
    tr_i, va_i = next(sss2.split(np.zeros(len(dev)), dev.label))
    tr, va, te = dev.loc[tr_i], dev.loc[va_i], s.loc[te_i]
    shared = len(set(te.group) & set(tr.group))
    SCV["random_split_test_clusters_also_in_train"] = int(shared); SCV["random_split_test_clusters"] = int(te.group.nunique())
    print(f"random split: {shared} of {te.group.nunique()} test cluster images also have cells in training")
    m, _ = _train_eval(tr, va, te, "random_cell_split")
    del m; torch.cuda.empty_cache()


_sc_section("A data checks", part_a_checks)
_sc_section("B cluster-grouped classifier", part_b_grouped)
if RUN_LEAKY_REFERENCE:
    _sc_section("C random cell-level split (reference)", part_c_leaky)
json.dump(SCV, open(f"{SC}/sipakmed_check_values.json", "w"), indent=1, default=float)
with zipfile.ZipFile(f"{OUT}/sipakmed_check.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob(f"{SC}/*")):
        if not f.endswith(".pth"):
            z.write(f, "sipakmed_check/" + os.path.basename(f))
log(f"sipakmed_check: finished -> {OUT}/sipakmed_check.zip  (download this file and send it back)")
