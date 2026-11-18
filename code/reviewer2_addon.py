# %% [17] Reviewer-2 analyses (run after cells [1]-[16]; uses the saved per-image predictions)
# Produces, without retraining (CPU, a few minutes):
#   A  extended aggregation ablation incl. majority voting and entropy-weighted LOGIT aggregation  (R2.4)
#   B  entropy distributions of correct vs. incorrect predictions and risk-coverage curves           (R2.3)
#   C  per-class sensitivity / specificity / PPV / NPV / AUC with 95% bootstrap CIs, MCC             (R2.17, R2.18)
#   D  one-vs-rest ROC and precision-recall curves (in-domain, cross-domain, few-shot)               (minor 15)
#   E  quantitative feature alignment: MMD, CORAL distance, domain-classifier accuracy               (R2.25)
#   F  failure analysis: high-confidence errors, image statistics per dataset, SIPaKMeD conf. matrix (R2.7, R2.30)
# and, optionally (GPU, ~30-45 min):
#   G  repeated stratified 25/75 Malhari splits for the source-only vs. few-shot comparison           (R2.13)
# Everything is written to OUT/review2/ and zipped into OUT/review2_bundle.zip.
import traceback
from sklearn.metrics import (matthews_corrcoef, roc_curve, precision_recall_curve, average_precision_score,
                             roc_auc_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

RUN_REPEATED_SPLITS = True        # part G (needs GPU time, ~30-45 min); set False to skip
N_REPEATED_SPLITS = 5
R2 = f"{OUT}/review2"; os.makedirs(R2, exist_ok=True)
R2V = {}                          # values for the manuscript / response letter


def _section(name, fn):
    t0 = time.time()
    try:
        fn(); log(f"review2: {name} done ({time.time() - t0:.0f} s)")
    except Exception:
        log(f"review2: {name} FAILED"); traceback.print_exc()


def _seeds(exp):
    return [s for s in range(5) if load_seed(exp, s) is not None]


def _member_logits(exp, seed, evalset):
    R = load_seed(exp, seed)
    if R is None or f"{evalset}__logits" not in R[0]:
        return None
    y = R[0][f"{evalset}__y"]
    Zs = [r[f"{evalset}__logits"] for r in R]
    temps = [fit_temperature(r["val__logits"], r["val__y"]) for r in R]
    return y, Zs, temps, R


def _rules(Zs, temps):
    """all aggregation rules compared in the ablation (probabilities returned for every rule)"""
    k = Zs[0].shape[1]
    P = [softmax(z, 1) for z in Zs]
    out = {}
    votes = np.stack([p.argmax(1) for p in P])                             # majority voting
    vfrac = np.stack([(votes == c).mean(0) for c in range(k)], 1)
    tieb = np.mean(P, 0) * 1e-6                                            # ties broken by mean probability
    out["Majority voting"] = (vfrac + tieb) / (vfrac + tieb).sum(1, keepdims=True)
    out["Uniform probability averaging"] = aggregate(Zs, "uniform_prob")
    out["Uniform logit averaging"] = aggregate(Zs, "uniform_logit")
    out["Entropy-weighted probability"] = aggregate(Zs, "entropy_weight")
    Zt = np.stack([z / t for z, t in zip(Zs, temps)])
    Pt = softmax(Zt, axis=2)
    H = np.stack([entropy(p, True) for p in Pt])
    W = softmax(-H / CFG["TAU"], axis=0)
    out["Entropy-weighted logit + TS"] = softmax((W[..., None] * Zt).sum(0), axis=1)
    out["Entropy-weighted probability + TS (proposed)"] = aggregate(Zs, "entropy_weight", temps)
    return out


# ------------------------------------------------------------------ A  aggregation ablation
def part_a():
    rows = []
    for evalset, lab in [("lbc_test", "LBC test"), ("mal_tt", "Malhari target-test")]:
        res = defaultdict(list)
        # single networks: EfficientNet-B3 without attention and EfficientNet-B3 + PCBAM
        for exp, name in [("base_effb3_nocbam", "Single EfficientNet-B3 (no attention)"),
                          ("lbc_in", "Single EfficientNet-B3 + PCBAM")]:
            for s in _seeds(exp):
                d = _member_logits(exp, s, evalset)
                if d is None:
                    continue
                y, Zs, _, _ = d
                for z in Zs:
                    res[name].append(metrics(y, softmax(z, 1), BETHESDA))
        for s in _seeds("lbc_in"):
            y, Zs, temps, _ = _member_logits("lbc_in", s, evalset)
            for name, p in _rules(Zs, temps).items():
                res[name].append(metrics(y, p, BETHESDA))
        order = ["Single EfficientNet-B3 (no attention)", "Single EfficientNet-B3 + PCBAM", "Majority voting",
                 "Uniform probability averaging", "Uniform logit averaging", "Entropy-weighted probability",
                 "Entropy-weighted logit + TS", "Entropy-weighted probability + TS (proposed)"]
        for name in order:
            if not res[name]:
                continue
            r = {"set": lab, "rule": name, "n": len(res[name])}
            for m in ["accuracy", "balanced_accuracy", "f1", "auc", "ece", "brier", "nll", "aurc", "err_auroc"]:
                v = np.array([x[m] for x in res[name]], float)
                r[m + "_mean"], r[m + "_sd"] = np.nanmean(v), (np.nanstd(v, ddof=1) if np.sum(~np.isnan(v)) > 1 else 0.0)
            rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(f"{R2}/table_ablation.csv", index=False)
    with open(f"{R2}/tab_ablation.tex", "w") as fh:
        cur = None
        for _, r in df.iterrows():
            if r["set"] != cur:
                if cur is not None:
                    fh.write("\\midrule\n")
                fh.write(f"\\multicolumn{{7}}{{l}}{{\\textit{{{r['set']}}}}} \\\\\n"); cur = r["set"]
            f = lambda m, pct=False, nd=3: (f"{100*r[m+'_mean']:.2f}" if pct else f"{r[m+'_mean']:.{nd}f}")
            fh.write(f"{r['rule']} & {f('accuracy', True)} & {f('f1', nd=4)} & {f('auc', nd=4)} & {f('ece')} & "
                     f"{f('brier')} & {f('err_auroc', nd=2)} \\\\\n")
    R2V["ablation"] = df.to_dict(orient="records")
    print(df[["set", "rule", "n", "accuracy_mean", "f1_mean", "auc_mean", "ece_mean", "err_auroc_mean"]].to_string())


# ------------------------------------------------------------------ B  uncertainty figures
def part_b():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for j, (evalset, lab) in enumerate([("lbc_test", "LBC test"), ("mal_tt", "Malhari target-test")]):
        Hc, Hw = [], []
        for s in _seeds("lbc_in"):
            y, Zs, temps, _ = _member_logits("lbc_in", s, evalset)
            p = aggregate(Zs, "entropy_weight", temps); h = entropy(p, True); ok = p.argmax(1) == y
            Hc += list(h[ok]); Hw += list(h[~ok])
        bins = np.linspace(0, 1, 26)
        ax[j].hist(Hc, bins=bins, alpha=0.6, density=True, label=f"correct (n={len(Hc)})", color="#2b7bba")
        ax[j].hist(Hw, bins=bins, alpha=0.6, density=True, label=f"incorrect (n={len(Hw)})", color="#d7301f")
        ax[j].set_xlabel("normalised ensemble entropy"); ax[j].set_ylabel("density"); ax[j].legend()
        ax[j].set_title(f"{lab}: entropy of correct vs. incorrect")
        if len(Hw) and len(Hc):
            R2V[f"entropy_{evalset}"] = {"median_correct": float(np.median(Hc)), "median_incorrect": float(np.median(Hw)),
                                         "mannwhitney_p": float(stats.mannwhitneyu(Hw, Hc, alternative="greater").pvalue)}
    # risk-coverage on Malhari, seed 0
    y, Zs, temps, _ = _member_logits("lbc_in", _seeds("lbc_in")[0], "mal_tt")
    curves = {"Single model (fold 0)": softmax(Zs[0], 1), "Uniform probability averaging": aggregate(Zs, "uniform_prob"),
              "Entropy-weighted + TS (proposed)": aggregate(Zs, "entropy_weight", temps)}
    for name, p in curves.items():
        order = np.argsort(entropy(p)); err = (p.argmax(1) != y).astype(float)[order]
        cov = np.arange(1, len(err) + 1) / len(err); risk = np.cumsum(err) / np.arange(1, len(err) + 1)
        ax[2].plot(cov, risk, label=f"{name} (AURC {risk.mean():.3f})")
    ax[2].set_xlabel("coverage (fraction of images retained)"); ax[2].set_ylabel("error rate on retained images")
    ax[2].set_title("Risk-coverage, Malhari target-test"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{R2}/uncertainty_analysis.pdf"); plt.savefig(f"{R2}/uncertainty_analysis.png", dpi=200)
    plt.close()


# ------------------------------------------------------------------ C  clinical metrics with CIs
def _clin(y, p, classes, nboot=1000, seed=0):
    yh = p.argmax(1); k = len(classes); rng = np.random.default_rng(seed)
    def stat(yy, pp):
        hh = pp.argmax(1); o = {}
        for c in range(k):
            tp = ((hh == c) & (yy == c)).sum(); fn = ((hh != c) & (yy == c)).sum()
            fp = ((hh == c) & (yy != c)).sum(); tn = ((hh != c) & (yy != c)).sum()
            o[f"sens_{c}"] = tp / (tp + fn) if tp + fn else np.nan
            o[f"spec_{c}"] = tn / (tn + fp) if tn + fp else np.nan
            o[f"ppv_{c}"] = tp / (tp + fp) if tp + fp else np.nan
            o[f"npv_{c}"] = tn / (tn + fn) if tn + fn else np.nan
            try:
                o[f"auc_{c}"] = roc_auc_score((yy == c).astype(int), pp[:, c])
            except ValueError:
                o[f"auc_{c}"] = np.nan
        o["mcc"] = matthews_corrcoef(yy, hh)
        ab = (yy != 0).astype(int); abh = (hh != 0).astype(int)            # screening: abnormal vs NILM
        o["screen_sens"] = (abh[ab == 1] == 1).mean() if ab.sum() else np.nan
        o["screen_spec"] = (abh[ab == 0] == 0).mean() if (ab == 0).sum() else np.nan
        return o
    point = stat(y, p)
    idx = [np.where(y == c)[0] for c in np.unique(y)]
    boots = []
    for _ in range(nboot):
        ix = np.concatenate([rng.choice(i, len(i)) for i in idx]); boots.append(stat(y[ix], p[ix]))
    ci = {key: tuple(np.nanpercentile([b[key] for b in boots], [2.5, 97.5])) for key in point}
    return point, ci


def part_c():
    rows, tex = [], []
    for exp, evalset, lab in [("lbc_in", "lbc_test", "LBC test (in-domain)"),
                              ("lbc_in", "mal_tt", "Malhari target-test (LBC only)"),
                              ("lbc_in_fs", "mal_tt", "Malhari target-test (few-shot)")]:
        if not _seeds(exp):
            continue
        y, Zs, temps, _ = _member_logits(exp, _seeds(exp)[0], evalset)
        p = aggregate(Zs, "entropy_weight", temps)
        point, ci = _clin(y, p, BETHESDA)
        tex.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{lab}}}; MCC {point['mcc']:.3f} [{ci['mcc'][0]:.3f}, {ci['mcc'][1]:.3f}]; "
                   f"screening sensitivity {point['screen_sens']:.3f}, specificity {point['screen_spec']:.3f}}} \\\\")
        for c, cname in enumerate(BETHESDA):
            r = {"set": lab, "class": cname, "n": int((y == c).sum())}
            cells = [cname + f" ({r['n']})"]
            for m in ["sens", "spec", "ppv", "npv", "auc"]:
                v, (lo, hi) = point[f"{m}_{c}"], ci[f"{m}_{c}"]
                r[m], r[m + "_lo"], r[m + "_hi"] = v, lo, hi
                cells.append("--" if v != v else f"{v:.3f} [{lo:.2f}, {hi:.2f}]")
            rows.append(r); tex.append(" & ".join(cells) + " \\\\")
        rows.append({"set": lab, "class": "ALL", "mcc": point["mcc"], "mcc_lo": ci["mcc"][0], "mcc_hi": ci["mcc"][1],
                     "screen_sens": point["screen_sens"], "screen_spec": point["screen_spec"]})
        tex.append("\\midrule")
    pd.DataFrame(rows).to_csv(f"{R2}/table_clinical.csv", index=False)
    open(f"{R2}/tab_clinical.tex", "w").write("\n".join(tex[:-1]) + "\n")
    R2V["clinical"] = rows


# ------------------------------------------------------------------ D  ROC and PR curves
def part_d():
    sets = [("lbc_in", "lbc_test", "LBC test"), ("lbc_in", "mal_tt", "Malhari (LBC only)"),
            ("lbc_in_fs", "mal_tt", "Malhari (few-shot)")]
    sets = [s for s in sets if _seeds(s[0])]
    fig, ax = plt.subplots(2, len(sets), figsize=(5 * len(sets), 9))
    ax = np.array(ax).reshape(2, -1)
    for j, (exp, evalset, lab) in enumerate(sets):
        y, Zs, temps, _ = _member_logits(exp, _seeds(exp)[0], evalset)
        p = aggregate(Zs, "entropy_weight", temps)
        for c, cname in enumerate(BETHESDA):
            yb = (y == c).astype(int)
            if 0 < yb.sum() < len(yb):
                fpr, tpr, _ = roc_curve(yb, p[:, c]); auc_c = roc_auc_score(yb, p[:, c])
                ax[0, j].plot(fpr, tpr, label=f"{cname} (AUC {auc_c:.3f})")
                pr, rc, _ = precision_recall_curve(yb, p[:, c]); ap = average_precision_score(yb, p[:, c])
                ax[1, j].plot(rc, pr, label=f"{cname} (AP {ap:.3f})")
        ax[0, j].plot([0, 1], [0, 1], "k--", lw=0.8)
        ax[0, j].set_title(f"ROC (one-vs-rest): {lab}"); ax[0, j].set_xlabel("1 - specificity"); ax[0, j].set_ylabel("sensitivity")
        ax[1, j].set_title(f"Precision-recall: {lab}"); ax[1, j].set_xlabel("recall"); ax[1, j].set_ylabel("precision")
        ax[0, j].legend(fontsize=8, loc="lower right"); ax[1, j].legend(fontsize=8, loc="lower left")
    plt.tight_layout(); plt.savefig(f"{R2}/roc_pr_curves.pdf"); plt.savefig(f"{R2}/roc_pr_curves.png", dpi=200); plt.close()


# ------------------------------------------------------------------ E  quantitative alignment
def _rbf_mmd2(X, Y, rng):
    n = min(len(X), len(Y), 400)
    X = X[rng.choice(len(X), n, replace=False)]; Y = Y[rng.choice(len(Y), n, replace=False)]
    Z = np.vstack([X, Y]); d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    sig2 = np.median(d2[d2 > 0])
    K = np.exp(-d2 / sig2)
    kxx, kyy, kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) - 2 * kxy.mean())


def _coral_dist(X, Y):
    d = X.shape[1]
    return float(((np.cov(X, rowvar=False) - np.cov(Y, rowvar=False)) ** 2).sum() / (4 * d * d))


def _domain_acc(X, Y):
    Z = np.vstack([X, Y]); t = np.r_[np.zeros(len(X)), np.ones(len(Y))]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    return float(cross_val_score(clf, Z, t, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                 scoring="balanced_accuracy").mean())


def _knn_target_acc(X, ys, Y, yt):
    """class-structure transfer: nearest-centroid of source classes applied to target features"""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8); Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    cents = np.stack([Xn[ys == c].mean(0) for c in range(len(BETHESDA))])
    return float(balanced_accuracy_score(yt, (Yn @ cents.T).argmax(1)))


def part_e():
    rng = np.random.default_rng(0); rows = []
    for exp, name in [("lbc_in", "Source only (LBC)"), ("lbc_in_da", "CORAL + MK-MMD"),
                      ("lbc_in_fs", "Few-shot (labelled Malhari)")]:
        vals = defaultdict(list)
        for s in _seeds(exp):
            R = load_seed(exp, s)
            for r in R:
                X, ys = r["lbc_test__feats"], r["lbc_test__y"]; Y, yt = r["mal_tt__feats"], r["mal_tt__y"]
                vals["mmd2"].append(_rbf_mmd2(X, Y, rng)); vals["coral"].append(_coral_dist(X, Y))
                vals["domain_acc"].append(_domain_acc(X, Y)); vals["centroid_bal_acc"].append(_knn_target_acc(X, ys, Y, yt))
        if vals:
            rows.append({"model": name, "n_networks": len(vals["mmd2"]),
                         **{f"{k}_mean": float(np.mean(v)) for k, v in vals.items()},
                         **{f"{k}_sd": float(np.std(v, ddof=1)) for k, v in vals.items()}})
    df = pd.DataFrame(rows); df.to_csv(f"{R2}/table_alignment.csv", index=False)
    with open(f"{R2}/tab_alignment.tex", "w") as fh:
        for _, r in df.iterrows():
            fh.write(f"{r['model']} & {r['n_networks']} & {r['mmd2_mean']:.4f} $\\pm$ {r['mmd2_sd']:.4f} & "
                     f"{r['coral_mean']:.2e} & {100*r['domain_acc_mean']:.1f} $\\pm$ {100*r['domain_acc_sd']:.1f} & "
                     f"{100*r['centroid_bal_acc_mean']:.1f} $\\pm$ {100*r['centroid_bal_acc_sd']:.1f} \\\\\n")
    R2V["alignment"] = rows
    print(df.to_string())


# ------------------------------------------------------------------ F  failure analysis
def _img_stats(paths, n=150, seed=0):
    rng = np.random.default_rng(seed); paths = list(paths)
    sel = rng.choice(len(paths), min(n, len(paths)), replace=False); out = []
    for i in sel:
        im = Image.open(paths[i]).convert("RGB"); w, h = im.size
        a = np.asarray(im.resize((512, 512))).astype(float)
        g = a.mean(2); lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        hsv = np.asarray(im.resize((256, 256)).convert("HSV")).astype(float)
        out.append({"width": w, "height": h, "brightness": g.mean(), "contrast": g.std(),
                    "R": a[..., 0].mean(), "G": a[..., 1].mean(), "B": a[..., 2].mean(),
                    "saturation": hsv[..., 1].mean(), "sharpness": lap.var()})
    return pd.DataFrame(out)


def part_f():
    # image statistics per dataset (staining / illumination / resolution / sharpness differences)
    st = []
    for name, df in [("Mendeley LBC", SPL_LBC), ("Malhari", SPL_MAL)]:
        s = _img_stats(df.path.values); st.append(s.median().rename(name))
        R2V[f"imgstats_{name}"] = s.median().to_dict()
    stdf = pd.concat(st, axis=1).T; stdf.to_csv(f"{R2}/table_image_statistics.csv")
    print(stdf.round(2).to_string())
    with open(f"{R2}/tab_image_statistics.tex", "w") as fh:
        for name, r in stdf.iterrows():
            fh.write(f"{name} & {int(r.width)}$\\times${int(r.height)} & {r.brightness:.1f} & {r.contrast:.1f} & "
                     f"{r.R:.0f}/{r.G:.0f}/{r.B:.0f} & {r.saturation:.1f} & {r.sharpness:.1f} \\\\\n")
    # high-confidence errors on Malhari (source-only ensemble, seed 0)
    s0 = _seeds("lbc_in")[0]
    y, Zs, temps, R = _member_logits("lbc_in", s0, "mal_tt")
    p = aggregate(Zs, "entropy_weight", temps); yh = p.argmax(1); conf = p.max(1)
    paths = R[0]["mal_tt__path"]; wrong = np.where(yh != y)[0]
    top = wrong[np.argsort(-conf[wrong])][:12]
    fig, ax = plt.subplots(2, 6, figsize=(18, 6.6))
    for a, i in zip(ax.ravel(), top):
        a.imshow(Image.open(paths[i]).convert("RGB")); a.axis("off")
        a.set_title(f"true {BETHESDA[y[i]]} | pred {BETHESDA[yh[i]]}\nconf {conf[i]:.2f}", fontsize=9)
    for a in ax.ravel()[len(top):]:
        a.axis("off")
    plt.tight_layout(); plt.savefig(f"{R2}/failure_cases.png", dpi=150); plt.savefig(f"{R2}/failure_cases.pdf"); plt.close()
    cm = confusion_matrix(y, yh, labels=range(4))
    R2V["failure"] = {"n_errors": int(len(wrong)), "n": int(len(y)),
                      "errors_by_true_class": {BETHESDA[c]: int(((y == c) & (yh != y)).sum()) for c in range(4)},
                      "abnormal_called_nilm": int(((y != 0) & (yh == 0)).sum()), "n_abnormal": int((y != 0).sum()),
                      "high_conf_errors_conf_gt_0.9": int((conf[wrong] > 0.9).sum()),
                      "mean_conf_correct": float(conf[yh == y].mean()), "mean_conf_wrong": float(conf[wrong].mean()),
                      "cm": cm.tolist()}
    print(R2V["failure"])
    # SIPaKMeD held-out confusion matrix with exact error counts (R2.7)
    fp = f"{run_dir('sip_pre', 0, 0)}/preds.npz"
    if os.path.exists(fp):
        d = dict(np.load(fp, allow_pickle=True)); ys, zs = d["sip_test__y"], d["sip_test__logits"]
        cms = confusion_matrix(ys, zs.argmax(1), labels=range(len(SIP_CLASSES)))
        names = [c.replace("im_", "") for c in SIP_CLASSES]
        fig, a = plt.subplots(figsize=(6.5, 5.5)); a.imshow(cms, cmap="Blues")
        for i in range(len(names)):
            for j in range(len(names)):
                a.text(j, i, cms[i, j], ha="center", va="center", color="white" if cms[i, j] > cms.max() / 2 else "black")
        a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=30, ha="right"); a.set_yticks(range(len(names)))
        a.set_yticklabels(names); a.set_xlabel("Predicted"); a.set_ylabel("True"); a.set_title("SIPaKMeD held-out test cells")
        plt.tight_layout(); plt.savefig(f"{R2}/sipakmed_cm.png", dpi=200); plt.savefig(f"{R2}/sipakmed_cm.pdf"); plt.close()
        R2V["sipakmed_cm"] = {"cm": cms.tolist(), "errors": int(len(ys) - np.trace(cms)), "n": int(len(ys))}


# ------------------------------------------------------------------ G  repeated Malhari splits
def part_g():
    allm = SPL_MAL.copy().reset_index(drop=True); allm["strat"] = allm.label
    res = []
    for k in range(N_REPEATED_SPLITS):
        sd = 1000 + k
        tt_i, ad_i = sgkf_split(allm, int(round(1 / CFG["MAL_ADAPT_FRAC"])), sd)[0]
        tt, ad = allm.loc[tt_i], allm.loc[ad_i].reset_index(drop=True)
        fs_tr_i, fs_va_i = sgkf_split(ad, 5, sd + 1)[0]
        fs_tr, fs_va = ad.loc[fs_tr_i], ad.loc[fs_va_i]
        warm_cache(allm.path)
        ytt = np.array([BETHESDA.index(l) for l in tt.label]); yva = np.array([BETHESDA.index(l) for l in fs_va.label])
        Z_src, T_src, Z_fs, T_fs = [], [], [], []
        for fold in range(CFG["N_FOLDS"]):
            base = load_base("imagenet", 0, fold)
            z, _ = predict(base, tt, BETHESDA); Z_src.append(z)
            rb = dict(np.load(f"{run_dir('lbc_in', 0, fold)}/preds.npz", allow_pickle=True))
            T_src.append(fit_temperature(rb["val__logits"], rb["val__y"]))
            seed_all(sd * 10 + fold)
            m, _ = fit(base, fs_tr, fs_va, BETHESDA, epochs=CFG["FS_EPOCHS"], lr=CFG["FS_LR"],
                       patience=CFG["FS_PATIENCE"], bs=CFG["FS_BS"], min_epochs=CFG["FS_MIN_EPOCHS"])
            z, _ = predict(m, tt, BETHESDA); Z_fs.append(z)
            zv, _ = predict(m, fs_va, BETHESDA); T_fs.append(fit_temperature(zv, yva))
            del m, base; torch.cuda.empty_cache()
        ms_ = metrics(ytt, aggregate(Z_src, "entropy_weight", T_src), BETHESDA)
        mf_ = metrics(ytt, aggregate(Z_fs, "entropy_weight", T_fs), BETHESDA)
        res.append({"split": k, "n_target_test": len(tt),
                    **{f"src_{m}": ms_[m] for m in ["accuracy", "balanced_accuracy", "f1", "auc"]},
                    **{f"fs_{m}": mf_[m] for m in ["accuracy", "balanced_accuracy", "f1", "auc"]}})
        log(f"review2 G: split {k}: source-only acc {ms_['accuracy']:.4f} F1 {ms_['f1']:.4f} | "
            f"few-shot acc {mf_['accuracy']:.4f} F1 {mf_['f1']:.4f}")
        pd.DataFrame(res).to_csv(f"{R2}/table_repeated_splits.csv", index=False)
    df = pd.DataFrame(res); summ = {}
    for m in ["accuracy", "balanced_accuracy", "f1", "auc"]:
        a, b = df[f"fs_{m}"].values, df[f"src_{m}"].values
        summ[m] = {"src_mean": float(b.mean()), "src_sd": float(b.std(ddof=1)), "fs_mean": float(a.mean()),
                   "fs_sd": float(a.std(ddof=1)), "diff": float((a - b).mean()),
                   "t_p": float(stats.ttest_rel(a, b).pvalue), "n_splits": len(a)}
    R2V["repeated_splits"] = summ
    with open(f"{R2}/tab_repeated_splits.tex", "w") as fh:
        for m, lab in [("accuracy", "Accuracy (\\%)"), ("balanced_accuracy", "Balanced accuracy (\\%)"),
                       ("f1", "Macro-F1"), ("auc", "Macro-AUC")]:
            s = summ[m]; f = 100 if "acc" in m else 1; nd = 2 if f == 100 else 4
            fh.write(f"{lab} & {f*s['src_mean']:.{nd}f} $\\pm$ {f*s['src_sd']:.{nd}f} & {f*s['fs_mean']:.{nd}f} $\\pm$ "
                     f"{f*s['fs_sd']:.{nd}f} & {f*s['diff']:+.{nd}f} & {s['t_p']:.3f} \\\\\n")
    print(summ)


for _name, _fn in [("A ablation", part_a), ("B uncertainty", part_b), ("C clinical metrics", part_c),
                   ("D ROC/PR", part_d), ("E alignment", part_e), ("F failure analysis", part_f)]:
    _section(_name, _fn)
if RUN_REPEATED_SPLITS:
    _section("G repeated Malhari splits", part_g)

json.dump(R2V, open(f"{R2}/review2_values.json", "w"), indent=1, default=float)
with zipfile.ZipFile(f"{OUT}/review2_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob(f"{R2}/*")):
        z.write(f, "review2/" + os.path.basename(f))
log(f"review2: finished -> {OUT}/review2_bundle.zip  (download this file and send it back)")
