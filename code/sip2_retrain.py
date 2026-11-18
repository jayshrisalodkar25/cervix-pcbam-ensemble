# %% [19] Re-train every SIPaKMeD-initialised model from the fully trained SIPaKMeD network of cell [18]
# ------------------------------------------------------------------------------------------------------------
# Run after cells [1]-[16] ("Save & Run All"). Delete cells [17] and [18] first (their results are already in
# the manuscript; re-running [18] would train a slightly different SIPaKMeD network).
#
# What it does
#   0  finds the SIPaKMeD network of cell [18] (trained on the 2900 SIPaKMeD development cells only, selected on
#      the 336 validation cells), verifies it on the 813 held-out cells and stores it as sipakmed_pretrained_v2.pth
#   1  re-trains, with exactly the original code, hyper-parameters and seeds, the four SIPaKMeD-initialised
#      configurations; the only change is the initial checkpoint:
#         lbc_sip2     LBC models, 3 seeds x 5 folds          (= lbc_sip)
#         lbc_sip2_da  + CORAL/MK-MMD, 3 x 5                  (= lbc_sip_da)
#         lbc_sip2_fs  + few-shot Malhari, 3 x 5              (= lbc_sip_fs)
#         merged_sip2  multi-domain, 1 seed x 5 folds         (= merged_sip)
#      Seeds, folds, data and training order are identical, so every model is paired with its ImageNet and its
#      short-schedule SIPaKMeD counterpart.
#   2  evaluates the ensembles exactly as cell [13] and writes drop-in versions of Table 7 (tab_main) and Table 8
#      (tab_da) in which the SIPaKMeD rows come from the new checkpoint, plus a direct old-vs-new comparison.
# Output: revision_outputs/sip2/ and revision_outputs/sip2_bundle.zip (download and send back).
# Time on a T4/P100: about 3-3.5 h. Resumable: if the session ends, attach this notebook's output as input and
# run again - finished models are skipped and the same checkpoint is reused.
# ------------------------------------------------------------------------------------------------------------
import gc, traceback

RUN_SIP2_TRAINING = True            # False = only evaluate the models that are already finished
SIP2_DIR = f"{OUT}/sip2"; os.makedirs(SIP2_DIR, exist_ok=True)
SIP2_CKPT = f"{OUT}/sipakmed_pretrained_v2.pth"          # top-level .pth -> copied by resume_previous()
SIP2_SOURCE_NAME = "sipakmed_cluster_grouped_v2.pth"      # written by cell [18]
SIP2_REF_ACC = 0.9053                                     # cell [18] result on the held-out cells (flip TTA)
S2V = {"config": {"MIN_EPOCHS": CFG["MIN_EPOCHS"], "FS_MIN_EPOCHS": CFG["FS_MIN_EPOCHS"], "SEEDS_MAIN": CFG["SEEDS_MAIN"],
                  "SEEDS_OTHER": CFG["SEEDS_OTHER"], "LR": CFG["LR"], "MAX_EPOCHS": CFG["MAX_EPOCHS"]}}
EXP_MAP = {"lbc_sip": "lbc_sip2", "lbc_sip_da": "lbc_sip2_da", "lbc_sip_fs": "lbc_sip2_fs", "merged_sip": "merged_sip2"}


def _md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def s2_predict_tta(model, df, classes):
    """mean softmax over the image and its horizontal / vertical flips (identical to cell [18])"""
    model.eval(); ps = []
    with torch.no_grad():
        for x, _ in loader(df, classes, train=False, shuffle=False):
            x = x.to(DEVICE)
            with autocast():
                p = sum(torch.softmax(model(v).float(), 1) for v in (x, torch.flip(x, [3]), torch.flip(x, [2]))) / 3
            ps.append(p.cpu())
    return torch.cat(ps).numpy()


# ------------------------------------------------------------------ 0  checkpoint: find, verify, freeze
def s2_prepare_checkpoint():
    sip_tr = SPL_SIP[SPL_SIP.partition == "dev"]; sip_va = SPL_SIP[SPL_SIP.partition == "val"]
    sip_te = SPL_SIP[SPL_SIP.partition == "test"]
    if os.path.exists(SIP2_CKPT):
        src, how = SIP2_CKPT, "reused (earlier session of this cell)"
    else:
        # the copy in the attached previous output is the network whose results are reported in Table 5;
        # it is preferred over a copy written by a re-run of cell [18] in this session
        inp = sorted(glob.glob(f"{INPUT_ROOT}/**/{SIP2_SOURCE_NAME}", recursive=True))
        here = f"{OUT}/sipakmed_check/{SIP2_SOURCE_NAME}"
        print("cell [18] networks found in the inputs:", inp)
        if inp:
            src, how = inp[0], "cell [18] output (attached input)"
        elif os.path.exists(here):
            src, how = here, "cell [18] output (this session)"
        else:
            # fallback: train it again with exactly the recipe of cell [18]
            log("sip2: cell [18] network not found - training it with the cell [18] recipe (about 20-30 min)")
            warm_cache(SPL_SIP.path)
            seed_all(0)
            m = build_model("pcbam", len(SIP_CLASSES), "imagenet")
            m, info = fit(m, sip_tr, sip_va, SIP_CLASSES, epochs=40, lr=CFG["LR"], patience=CFG["PATIENCE"], bs=CFG["BS"],
                          loss="focal", schedule="onecycle", min_epochs=15)
            src, how = f"{SIP2_DIR}/{SIP2_SOURCE_NAME}", "re-trained in this cell (cell [18] recipe)"
            torch.save(m.state_dict(), src)
            S2V["retrained_info"] = {"epochs": info["epochs_run"], "best_val_f1": info["best_val_f1"]}
            del m; gc.collect(); torch.cuda.empty_cache() if CUDA else None
        shutil.copy(src, SIP2_CKPT)
    # verify on the held-out SIPaKMeD cells (never used for any decision)
    warm_cache(sip_te.path)
    m = DeepCervixModel(len(SIP_CLASSES), imagenet=False).to(DEVICE)
    m.load_state_dict(torch.load(SIP2_CKPT, map_location=DEVICE))
    y = np.array([SIP_CLASSES.index(l) for l in sip_te.label])
    p_tta = s2_predict_tta(m, sip_te, SIP_CLASSES)
    z, _ = predict(m, sip_te, SIP_CLASSES); p_plain = softmax(z, 1)
    r = metrics(y, p_tta, SIP_CLASSES)
    S2V["checkpoint"] = {"source": src, "how": how, "md5": _md5_file(SIP2_CKPT),
                         "sip_test_acc_tta": r["accuracy"], "sip_test_bal_acc_tta": r["balanced_accuracy"],
                         "sip_test_f1_tta": r["f1"], "sip_test_auc_tta": r["auc"],
                         "sip_test_acc_plain": accuracy_score(y, p_plain.argmax(1)), "n_test_cells": int(len(y))}
    del m; gc.collect(); torch.cuda.empty_cache() if CUDA else None
    log(f"sip2: checkpoint {how}: {src}")
    log(f"sip2: held-out SIPaKMeD cells: accuracy {100 * r['accuracy']:.2f}% (flip TTA), "
        f"{100 * S2V['checkpoint']['sip_test_acc_plain']:.2f}% (no TTA), macro-AUC {r['auc']:.4f}")
    assert r["accuracy"] > 0.80, "the SIPaKMeD checkpoint does not reproduce cell [18] - wrong file?"
    if not SMOKE and abs(r["accuracy"] - SIP2_REF_ACC) > 0.005:
        print(f"NOTE: accuracy differs from the cell [18] value {100 * SIP2_REF_ACC:.2f}% - Table 5 will be updated "
              f"with the value above.")


def build_sip2(num_classes):
    """same as build_model('pcbam', n, 'sipakmed') but from the new checkpoint (all tensors except fc)"""
    m = DeepCervixModel(num_classes, imagenet=False)
    sd = torch.load(SIP2_CKPT, map_location="cpu")
    own = m.state_dict()
    use = {k: v for k, v in sd.items() if k in own and not k.startswith("fc.") and v.shape == own[k].shape}
    n_expected = len([k for k in own if not k.startswith("fc.")])
    assert len(use) == n_expected, f"only {len(use)}/{n_expected} SIPaKMeD-v2 tensors matched"
    own.update(use); m.load_state_dict(own)
    return m.to(DEVICE)


# ------------------------------------------------------------------ 1  training tasks (copies of cell [10])
def s2_lbc(seed, fold):
    exp = "lbc_sip2"
    if is_done(exp, seed, fold) and os.path.exists(f"{run_dir(exp, seed, fold)}/model.pth"):
        return "skip"
    seed_all(seed * 1000 + fold)
    tr, va = lbc_fold(fold)
    m = build_sip2(4)
    m, info = fit(m, tr, va, BETHESDA, epochs=CFG["MAX_EPOCHS"], lr=CFG["LR"], patience=CFG["PATIENCE"],
                  min_epochs=CFG["MIN_EPOCHS"])
    save_run(exp, seed, fold, m, info, {"val": va, "lbc_test": LBC_TEST, "mal_tt": MAL_TT}, BETHESDA, keep_weights=True)
    return info


def s2_load_base(seed, fold):
    p = f"{run_dir('lbc_sip2', seed, fold)}/model.pth"
    if not os.path.exists(p):
        s2_lbc(seed, fold)
    m = build_model("pcbam", 4, "scratch")
    m.load_state_dict(torch.load(p, map_location=DEVICE))
    return m


def s2_uda(seed, fold):
    exp = "lbc_sip2_da"
    if is_done(exp, seed, fold):
        return "skip"
    seed_all(seed * 1000 + fold + 7)
    tr, va = lbc_fold(fold)
    m = s2_load_base(seed, fold)
    m, info = fit(m, tr, va, BETHESDA, epochs=CFG["UDA_EPOCHS"], lr=CFG["UDA_LR"], patience=999,
                  target_df=MAL_ADAPT, select="last")
    save_run(exp, seed, fold, m, info, {"val": va, "lbc_test": LBC_TEST, "mal_tt": MAL_TT}, BETHESDA)
    return info


def s2_fewshot(seed, fold):
    exp = "lbc_sip2_fs"
    if is_done(exp, seed, fold):
        return "skip"
    seed_all(seed * 1000 + fold + 13)
    m = s2_load_base(seed, fold)
    m, info = fit(m, MAL_FS_TR, MAL_FS_VA, BETHESDA, epochs=CFG["FS_EPOCHS"], lr=CFG["FS_LR"],
                  patience=CFG["FS_PATIENCE"], bs=CFG["FS_BS"], min_epochs=CFG["FS_MIN_EPOCHS"])
    save_run(exp, seed, fold, m, info, {"val": MAL_FS_VA, "lbc_test": LBC_TEST, "mal_tt": MAL_TT}, BETHESDA)
    return info


def s2_merged(seed, fold):
    exp = "merged_sip2"
    if is_done(exp, seed, fold):
        return "skip"
    seed_all(seed * 1000 + fold + 21)
    dev = SPL_MER[SPL_MER.partition == "dev"]
    tr, va = dev[dev.fold != fold], dev[dev.fold == fold]
    te = SPL_MER[SPL_MER.partition == "test"]
    m = build_sip2(4)
    m, info = fit(m, tr, va, BETHESDA, epochs=CFG["MAX_EPOCHS"], lr=CFG["LR"], patience=CFG["PATIENCE"],
                  min_epochs=CFG["MIN_EPOCHS"])
    save_run(exp, seed, fold, m, info, {"val": va, "merged_test": te}, BETHESDA)
    return info


S2_TASKS = {"lbc_sip2": s2_lbc, "lbc_sip2_da": s2_uda, "lbc_sip2_fs": s2_fewshot, "merged_sip2": s2_merged}


def s2_queue():
    """same order as the original queue: seed 0 (base, DA, few-shot per fold), multi-domain, then seeds 1, 2"""
    q, folds = [], range(CFG["N_FOLDS"])
    main_seeds = CFG["SEEDS_MAIN"]
    for f in folds:
        q += [("lbc_sip2", main_seeds[0], f), ("lbc_sip2_da", main_seeds[0], f), ("lbc_sip2_fs", main_seeds[0], f)]
    for s in CFG["SEEDS_OTHER"]:
        q += [("merged_sip2", s, f) for f in folds]
    for s in main_seeds[1:]:
        for f in folds:
            q += [("lbc_sip2", s, f), ("lbc_sip2_da", s, f), ("lbc_sip2_fs", s, f)]
    return q


S2_DUR = defaultdict(list)


def s2_run_queue():
    warm_cache(pd.concat([SPL_LBC.path, SPL_MAL.path]).unique())
    q = s2_queue(); n_done = 0
    for i, (exp, s, f) in enumerate(q):
        if is_done(exp, s, f) and (exp != "lbc_sip2" or os.path.exists(f"{run_dir(exp, s, f)}/model.pth")):
            n_done += 1
            continue
        est = np.median(S2_DUR[exp]) if S2_DUR[exp] else (20 * 60 if exp in ("lbc_sip2", "merged_sip2") else 8 * 60)
        if (time.time() - T_START) + est > CFG["MAX_HOURS"] * 3600:
            log(f"sip2: time budget reached - {len(q) - i} models left; attach the output and run again")
            return False
        t0 = time.time()
        r = S2_TASKS[exp](s, f)
        gc.collect(); torch.cuda.empty_cache() if CUDA else None
        n_done += 1
        if r == "skip":
            continue
        S2_DUR[exp].append(time.time() - t0)
        log(f"sip2 [{n_done}/{len(q)}] {exp} seed {s} fold {f}: {(time.time() - t0) / 60:.1f} min, "
            f"epochs {r['epochs_run']}, best val F1 {r['best_val_f1']:.4f}")
    log("sip2: all models finished")
    return True


# ------------------------------------------------------------------ 2  evaluation (identical to cell [13])
S2_RES = {}


def s2_get(exp, es, sub=None):
    k = (exp, es, sub)
    if k not in S2_RES:
        S2_RES[k] = {s: r for s in seeds_of(exp) for r in [ens_eval(exp, s, es, subset=sub)] if r}
    return S2_RES[k]                      # {seed: ens_eval result}


def s2_per_seed(exp, es, metric, sub=None, rule=MAIN if "MAIN" in globals() else "entropy_weight+TS"):
    return [r[rule][metric] for s, r in sorted(s2_get(exp, es, sub).items())]


RULE = "entropy_weight+TS"
ROWS = [  # identical to MAIN_ROWS of cell [13]
    ("EffNet-B3+PCBAM", "lbc_in", "lbc_test", None, "ImageNet", "LBC", "LBC (test)"),
    ("EffNet-B3+PCBAM", "lbc_sip", "lbc_test", None, "SIPaKMeD", "LBC", "LBC (test)"),
    ("EffNet-B3+PCBAM", "lbc_in", "mal_tt", None, "ImageNet", "LBC", "Malhari (75\\%)"),
    ("\\;+CORAL+MK-MMD", "lbc_in_da", "mal_tt", None, "ImageNet", "LBC + unlab.\\ Malhari 25\\%", "Malhari (75\\%)"),
    ("EffNet-B3+PCBAM", "lbc_sip", "mal_tt", None, "SIPaKMeD", "LBC", "Malhari (75\\%)"),
    ("\\;+CORAL+MK-MMD", "lbc_sip_da", "mal_tt", None, "SIPaKMeD", "LBC + unlab.\\ Malhari 25\\%", "Malhari (75\\%)"),
    ("EffNet-B3+PCBAM", "lbc_in_fs", "mal_tt", None, "ImageNet", "LBC $\\rightarrow$ lab.\\ Malhari 25\\%", "Malhari (75\\%)"),
    ("EffNet-B3+PCBAM", "lbc_sip_fs", "mal_tt", None, "SIPaKMeD", "LBC $\\rightarrow$ lab.\\ Malhari 25\\%", "Malhari (75\\%)"),
    ("EffNet-B3+PCBAM", "merged_in", "merged_test", "LBC", "ImageNet", "Multi-domain (LBC+Malhari)", "LBC (test)"),
    ("EffNet-B3+PCBAM", "merged_in", "merged_test", "Malhari", "ImageNet", "Multi-domain (LBC+Malhari)", "Malhari (test)"),
    ("EffNet-B3+PCBAM", "merged_in", "merged_test", None, "ImageNet", "Multi-domain (LBC+Malhari)", "Merged (test)"),
    ("EffNet-B3+PCBAM", "merged_sip", "merged_test", "LBC", "SIPaKMeD", "Multi-domain (LBC+Malhari)", "LBC (test)"),
    ("EffNet-B3+PCBAM", "merged_sip", "merged_test", "Malhari", "SIPaKMeD", "Multi-domain (LBC+Malhari)", "Malhari (test)"),
    ("EffNet-B3+PCBAM", "merged_sip", "merged_test", None, "SIPaKMeD", "Multi-domain (LBC+Malhari)", "Merged (test)"),
]
KEYS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"]


def s2_row(exp, es, sub):
    R = s2_get(exp, es, sub)
    if not R:
        return None
    v = {k: s2_per_seed(exp, es, k, sub) for k in KEYS}
    single = [np.mean([m["accuracy"] for m in r["single"]]) for s, r in sorted(R.items())]
    return v, len(R), float(np.mean(single))


def s2_tables():
    tex, csv, cmp_old_new = [], [], []
    for lab, exp, es, sub, pre, tr, te in ROWS:
        use = EXP_MAP.get(exp, exp)                     # SIPaKMeD rows -> new checkpoint
        res = s2_row(use, es, sub)
        if res is None:
            print(f"   (not finished: {use} {es} {sub})")
            continue
        v, n, single = res
        tex.append(f"{lab} & {pre} & {tr} & {te} & {ms(v['accuracy'], True)} & {ms(v['balanced_accuracy'], True)} & "
                   f"{ms(v['f1'])} & {ms(v['auc'])} \\\\")
        csv.append({"model": lab.replace("\\;", ""), "init": pre, "train": tr, "test": te, "seeds": n,
                    **{f"{k}_mean": np.mean(x) for k, x in v.items()},
                    **{f"{k}_sd": (np.std(x, ddof=1) if len(x) > 1 else 0) for k, x in v.items()},
                    "single_model_acc_mean": single})
        if exp in EXP_MAP:                               # direct old-vs-new comparison
            old = s2_row(exp, es, sub)
            if old:
                ov = old[0]
                cmp_old_new.append({"setting": f"{tr} -> {te}".replace("\\", ""), "exp_old": exp, "exp_new": use,
                                    **{f"{k}_old": np.mean(ov[k]) for k in KEYS}, **{f"{k}_new": np.mean(v[k]) for k in KEYS},
                                    "seeds_old": old[1], "seeds_new": n})
    open(f"{SIP2_DIR}/tab_main.tex", "w").write("\n".join(tex) + "\n")
    pd.DataFrame(csv).to_csv(f"{SIP2_DIR}/table_main.csv", index=False)
    pd.DataFrame(cmp_old_new).to_csv(f"{SIP2_DIR}/sip_old_vs_new.csv", index=False)
    rows = []
    for r in cmp_old_new:
        rows.append(f"{r['setting'].replace('%', '\\%')} & {100 * r['accuracy_old']:.2f} & {100 * r['accuracy_new']:.2f} & "
                    f"{100 * r['balanced_accuracy_old']:.2f} & {100 * r['balanced_accuracy_new']:.2f} & "
                    f"{r['f1_old']:.4f} & {r['f1_new']:.4f} & {r['auc_old']:.4f} & {r['auc_new']:.4f} \\\\")
    open(f"{SIP2_DIR}/tab_sip_old_vs_new.tex", "w").write("\n".join(rows) + "\n")
    return csv, cmp_old_new


def s2_compare(label, a, b, es, sub=None, level="model", rule_a=RULE, rule_b=RULE):
    """identical statistics to compare() of cell [13]; models are paired explicitly by (seed, fold)"""
    Ra, Rb = s2_get(a, es, sub), s2_get(b, es, sub)
    common = sorted(set(Ra) & set(Rb))
    if not common:
        return None
    if level == "model":
        res = {m: paired_tests([x[m] for s in common for x in Ra[s]["single"]],
                               [x[m] for s in common for x in Rb[s]["single"]]) for m in ["accuracy", "f1", "auc"]}
    else:
        res = {m: paired_tests([Ra[s][rule_a][m] for s in common], [Rb[s][rule_b][m] for s in common])
               for m in ["accuracy", "f1", "auc"]}
    for s in common:
        assert np.array_equal(Ra[s]["paths"], Rb[s]["paths"]), "different test images - cannot pair"
    mc = [mcnemar_exact(Ra[s]["y"], Ra[s]["probs"][rule_a].argmax(1), Rb[s]["probs"][rule_b].argmax(1)) for s in common]
    s0 = common[0]; y, pa, pb = Ra[s0]["y"], Ra[s0]["probs"][rule_a], Rb[s0]["probs"][rule_b]
    lo, hi = boot_ci(y, np.stack([pa, pb], 1), lambda t, q: f1_score(t, q[:, 0].argmax(1), average="macro") -
                                                         f1_score(t, q[:, 1].argmax(1), average="macro"))
    unit = "models" if level == "model" else "seeds"
    row = (f"{label} & {res['f1']['n']} {unit} & {100 * res['accuracy']['diff']:+.2f} ({res['accuracy']['t_p']:.3f}) & "
           f"{res['f1']['diff']:+.4f} ({res['f1']['t_p']:.3f}) & {res['auc']['diff']:+.4f} ({res['auc']['t_p']:.3f}) & "
           f"{res['f1']['wilcoxon_p']:.3f} & {holm_max(mc):.3f} \\\\")
    rec = {"comparison": label, "eval": es if sub is None else f"{es}:{sub}", "level": level, "mcnemar_p_per_seed": mc,
           "mcnemar_holm_max": holm_max(mc), "f1_diff_seed0_boot95": (lo, hi),
           **{f"{m}_{k}": v for m, d in res.items() for k, v in d.items()}}
    return row, rec


def s2_statistics():
    # Table 8 in the original order; the SIPaKMeD comparisons use the new checkpoint
    main_cmp = [("ImageNet: CORAL+MK-MMD vs base (Malhari)", "lbc_in_da", "lbc_in", "mal_tt", None, "model", RULE, RULE),
                ("SIPaKMeD: CORAL+MK-MMD vs base (Malhari)", "lbc_sip2_da", "lbc_sip2", "mal_tt", None, "model", RULE, RULE),
                ("ImageNet: few-shot vs base (Malhari)", "lbc_in_fs", "lbc_in", "mal_tt", None, "model", RULE, RULE),
                ("SIPaKMeD: few-shot vs base (Malhari)", "lbc_sip2_fs", "lbc_sip2", "mal_tt", None, "model", RULE, RULE),
                ("SIPaKMeD vs ImageNet init (LBC test)", "lbc_sip2", "lbc_in", "lbc_test", None, "model", RULE, RULE),
                ("SIPaKMeD vs ImageNet init (Malhari)", "lbc_sip2", "lbc_in", "mal_tt", None, "model", RULE, RULE),
                ("Entropy-weighted+TS vs uniform (LBC)", "lbc_in", "lbc_in", "lbc_test", None, "ensemble", RULE, "uniform_prob"),
                ("Entropy-weighted+TS vs uniform (Malhari)", "lbc_in", "lbc_in", "mal_tt", None, "ensemble", RULE, "uniform_prob")]
    extra_cmp = [("SIPaKMeD full vs short schedule (LBC test)", "lbc_sip2", "lbc_sip", "lbc_test", None),
                 ("SIPaKMeD full vs short schedule (Malhari)", "lbc_sip2", "lbc_sip", "mal_tt", None),
                 ("SIPaKMeD full vs short schedule, CORAL+MK-MMD (Malhari)", "lbc_sip2_da", "lbc_sip_da", "mal_tt", None),
                 ("SIPaKMeD full vs short schedule, few-shot (Malhari)", "lbc_sip2_fs", "lbc_sip_fs", "mal_tt", None),
                 ("SIPaKMeD vs ImageNet, CORAL+MK-MMD (Malhari)", "lbc_sip2_da", "lbc_in_da", "mal_tt", None),
                 ("SIPaKMeD vs ImageNet, few-shot (Malhari)", "lbc_sip2_fs", "lbc_in_fs", "mal_tt", None),
                 ("SIPaKMeD vs ImageNet, multi-domain (Malhari hold-out)", "merged_sip2", "merged_in", "merged_test", "Malhari"),
                 ("SIPaKMeD vs ImageNet, multi-domain (LBC hold-out)", "merged_sip2", "merged_in", "merged_test", "LBC"),
                 ("SIPaKMeD full vs short schedule, multi-domain (Malhari hold-out)", "merged_sip2", "merged_sip", "merged_test", "Malhari")]
    rows, recs, xrows, xrecs = [], [], [], []
    for lab, a, b, es, sub, lev, ra, rb in main_cmp:
        out = s2_compare(lab, a, b, es, sub, lev, ra, rb)
        if out:
            rows.append(out[0]); recs.append(out[1])
    for lab, a, b, es, sub in extra_cmp:
        out = s2_compare(lab, a, b, es, sub)
        if out:
            xrows.append(out[0]); xrecs.append(out[1])
    open(f"{SIP2_DIR}/tab_da.tex", "w").write("\n".join(rows) + "\n")
    pd.DataFrame(recs).to_csv(f"{SIP2_DIR}/table_statistics.csv", index=False)
    open(f"{SIP2_DIR}/tab_da_extra.tex", "w").write("\n".join(xrows) + "\n")
    pd.DataFrame(xrecs).to_csv(f"{SIP2_DIR}/table_statistics_extra.csv", index=False)
    return recs, xrecs


def s2_details():
    """per-class recalls, calibration, epochs and confusion matrices of the new models"""
    det = {}
    for exp, es, sub in [("lbc_sip2", "lbc_test", None), ("lbc_sip2", "mal_tt", None), ("lbc_sip2_da", "mal_tt", None),
                         ("lbc_sip2_fs", "mal_tt", None), ("merged_sip2", "merged_test", "Malhari"),
                         ("merged_sip2", "merged_test", "LBC"), ("lbc_sip", "mal_tt", None), ("lbc_sip_fs", "mal_tt", None),
                         ("lbc_in", "mal_tt", None), ("lbc_in_fs", "mal_tt", None)]:
        R = s2_get(exp, es, sub)
        if not R:
            continue
        k = f"{exp}|{es}|{sub}"
        det[k] = {m: float(np.mean([r[RULE][m] for r in R.values()])) for m in
                  ["accuracy", "balanced_accuracy", "f1", "auc", "ece", "brier", "nll", "err_auroc"] +
                  [f"recall_{c}" for c in BETHESDA]}
        det[k]["ece_uniform"] = float(np.mean([r["uniform_prob"]["ece"] for r in R.values()]))
        det[k]["single_acc"] = float(np.mean([np.mean([m["accuracy"] for m in r["single"]]) for r in R.values()]))
        det[k]["seeds"] = sorted(R)
        if "plot_cm" in globals() and exp.endswith("2"):
            r0 = R[sorted(R)[0]]
            plot_cm(r0[RULE]["cm"], BETHESDA, f"{SIP2_DIR}/cm_{exp}_{es}_{sub}.pdf", f"{exp} {es} {sub or ''}")
    ep = {}
    for exp in ["lbc_sip2", "lbc_sip2_fs", "merged_sip2", "lbc_sip", "lbc_sip_fs", "merged_sip", "lbc_in"]:
        logs = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/runs/{exp}/s*/f*/log.json"))]
        if logs:
            ep[exp] = {"n": len(logs), "epochs_mean": float(np.mean([l["epochs_run"] for l in logs])),
                       "epochs_min": int(min(l["epochs_run"] for l in logs)),
                       "best_val_f1_mean": float(np.mean([l["best_val_f1"] for l in logs])),
                       "best_val_f1_min": float(min(l["best_val_f1"] for l in logs))}
    return det, ep


def s2_main():
    _ok = True
    try:
        s2_prepare_checkpoint()
    except Exception:
        traceback.print_exc(); log("sip2: checkpoint step FAILED - nothing trained"); return
    if RUN_SIP2_TRAINING:
        try:
            S2V["all_models_finished"] = s2_run_queue()
        except Exception:
            traceback.print_exc(); log("sip2: training stopped by an error (finished models are kept)")
            S2V["all_models_finished"] = False
    S2V["seeds_finished"] = {e: seeds_of(e) for e in ["lbc_sip2", "lbc_sip2_da", "lbc_sip2_fs", "merged_sip2",
                                                     "lbc_sip", "lbc_in", "merged_in"]}
    for name, fn in [("tables", s2_tables), ("statistics", s2_statistics), ("details", s2_details)]:
        try:
            S2V[name] = fn()
        except Exception:
            traceback.print_exc(); log(f"sip2: {name} FAILED")
    json.dump(tidy(S2V) if "tidy" in globals() else S2V, open(f"{SIP2_DIR}/values_sip2.json", "w"), indent=1, default=str)
    # readable summary
    lines = ["SIPaKMeD-initialised models: short-schedule checkpoint (old) vs fully trained checkpoint (new)", "=" * 96,
             f"checkpoint: {S2V.get('checkpoint')}", f"all models finished: {S2V.get('all_models_finished')}",
             f"seeds finished: {S2V['seeds_finished']}", ""]
    if os.path.exists(f"{SIP2_DIR}/sip_old_vs_new.csv"):
        d = pd.read_csv(f"{SIP2_DIR}/sip_old_vs_new.csv")
        for _, r in d.iterrows():
            lines.append(f"{r.setting:58s} acc {100 * r.accuracy_old:6.2f} -> {100 * r.accuracy_new:6.2f} | "
                         f"bal {100 * r.balanced_accuracy_old:6.2f} -> {100 * r.balanced_accuracy_new:6.2f} | "
                         f"F1 {r.f1_old:.4f} -> {r.f1_new:.4f} | AUC {r.auc_old:.4f} -> {r.auc_new:.4f}")
    for f in ["table_statistics.csv", "table_statistics_extra.csv"]:
        if os.path.exists(f"{SIP2_DIR}/{f}"):
            d = pd.read_csv(f"{SIP2_DIR}/{f}"); lines.append("")
            for _, r in d.iterrows():
                lines.append(f"{r.comparison:62s} dAcc {100 * r.accuracy_diff:+6.2f} (p={r.accuracy_t_p:.3f}) | "
                             f"dF1 {r.f1_diff:+.4f} (p={r.f1_t_p:.3f}) | dAUC {r.auc_diff:+.4f} (p={r.auc_t_p:.3f}) | n={r.f1_n}")
    open(f"{SIP2_DIR}/SUMMARY.txt", "w").write("\n".join(lines)); print("\n".join(lines))
    with zipfile.ZipFile(f"{OUT}/sip2_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(glob.glob(f"{SIP2_DIR}/*")):
            if not f.endswith(".pth"):
                z.write(f, "sip2/" + os.path.basename(f))
    log(f"sip2: finished -> {OUT}/sip2_bundle.zip  (download this file and send it back)")


s2_main()
