# %% [markdown]
# # Major-revision experiments - cervical cytology (EfficientNet-B3 + Progressive CBAM)
#
# **Run this notebook top to bottom ("Save Version -> Save & Run All").** No edits are needed.
#
# * Settings: Accelerator = GPU (T4 or P100), Internet = ON (ImageNet weights).
# * Inputs already attached to your notebook are found automatically
#   (Mendeley LBC, Malhari, SIPaKMeD).
# * Everything is written to `/kaggle/working/revision_outputs/`.
# * The notebook stops training ~1 h before Kaggle's 12 h limit and always finishes
#   with the report cell. To continue in a new session, add this notebook's own output
#   ("Add Input" -> Notebook Output) and run again: finished models are skipped.
# * Upload `revision_outputs/manuscript_bundle.zip` contents into the manuscript folder
#   (`generated/`, `figures/`) and recompile the LaTeX.
#
# The model, loss, optimiser and augmentation are identical to the notebook used for
# the first submission (DeepCervixModel: CBAM after the 48/136/384-channel stages and
# on the final 1536-channel map, LayerNorm + 512/256 projection head, focal loss,
# AdamW + OneCycleLR, batch 32, 300 px, early stopping patience 7).
# What changed is the *protocol*: group-aware splits, an untouched test set, verified
# SIPaKMeD loading, real ensembling, multiple seeds, calibration and statistics.

# %% [1] Imports and configuration
import os, sys, json, time, math, random, hashlib, shutil, warnings, copy, glob, zipfile, itertools
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import (efficientnet_b3, EfficientNet_B3_Weights, resnet50, ResNet50_Weights,
                                densenet121, DenseNet121_Weights)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.manifold import TSNE
from scipy import stats
from scipy.special import softmax
from scipy.fft import dctn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None
T_START = time.time()
SMOKE = os.environ.get("CERVIX_SMOKE", "0") == "1"   # tiny self-test mode, never set on Kaggle

CFG = dict(
    SEEDS_MAIN=[0, 1, 2],          # seeds for LBC models, CORAL/MK-MMD and few-shot (editor point 3)
    SEEDS_OTHER=[0],               # seeds for multi-domain, cascade, baselines (budget)
    N_FOLDS=5, TEST_FRAC=0.20,     # 20 % untouched test set, 5 development folds
    MAL_ADAPT_FRAC=0.25,           # Malhari: 25 % adapt / 75 % target-test (as in the paper)
    MAL_MERGED_TEST_FRAC=0.20,     # Malhari hold-out for multi-domain training
    IMG=300, BS=32, LR=3e-4, WD=1e-4, MAX_EPOCHS=50, PATIENCE=7,      # = original notebook
    FOCAL_GAMMA=2.0, LABEL_SMOOTH=0.05,
    SIP_EPOCHS=15, SIP_LR=1e-4,                                       # = original pretraining
    UDA_EPOCHS=10, UDA_LR=1e-4, W_CORAL=1.0, W_MMD=1.0,
    FS_EPOCHS=30, FS_LR=1e-4, FS_PATIENCE=5, FS_BS=16,
    TAU=0.5,                       # entropy-weighting temperature, fixed a priori
    PHASH_HAMMING=4,               # near-duplicate threshold used for grouping
    EMB_COS=0.97,                  # embedding similarity: reported for visual check only
    BASELINES=["resnet50", "densenet121", "effb3_nocbam"],
    RUN_CASCADE=True, RUN_BASELINES=True,
    MAX_HOURS=11.0,                # stop starting new trainings after this (Kaggle limit 12 h)
    NUM_WORKERS=2, N_BOOT=2000,
)
if SMOKE:
    CFG.update(SEEDS_MAIN=[0, 1], IMG=64, BS=8, MAX_EPOCHS=2, PATIENCE=2, SIP_EPOCHS=1, UDA_EPOCHS=1,
               FS_EPOCHS=2, FS_BS=4, NUM_WORKERS=0, N_BOOT=50, MAX_HOURS=5)

INPUT_ROOT = os.environ.get("CERVIX_INPUT", "/kaggle/input")
OUT = os.environ.get("CERVIX_OUT", "/kaggle/working/revision_outputs")
for d in ["splits", "runs", "results", "generated", "figures", "audit"]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA = DEVICE.type == "cuda"
BETHESDA = ["NILM", "LSIL", "HSIL", "SCC"]
SIP_CLASSES = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial-Intermediate"]
print(f"device={DEVICE} | smoke={SMOKE} | out={OUT}")
if not CUDA and not SMOKE:
    print("WARNING: no GPU detected - set Accelerator = GPU in the notebook settings.")


def log(msg):
    el = (time.time() - T_START) / 3600
    print(f"[{el:5.2f} h] {msg}", flush=True)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %% [2] Resume from a previous session (copies earlier outputs if attached as input)
def resume_previous():
    found = []
    for root, dirs, files in os.walk(INPUT_ROOT):
        if os.path.basename(root) == "revision_outputs" and os.path.isdir(os.path.join(root, "runs")):
            found.append(root)
        if root.count(os.sep) - INPUT_ROOT.count(os.sep) > 4:
            dirs[:] = []
    for src in found:
        if os.path.abspath(src) == os.path.abspath(OUT):
            continue
        log(f"resuming from previous outputs: {src}")
        for sub in ["splits", "runs", "audit"]:
            if os.path.isdir(f"{src}/{sub}"):
                shutil.copytree(f"{src}/{sub}", f"{OUT}/{sub}", dirs_exist_ok=True)
        for f in glob.glob(f"{src}/*.pth"):
            if not os.path.exists(f"{OUT}/{os.path.basename(f)}"):
                shutil.copy(f, OUT)
    return found


RESUMED = resume_previous()


# %% [3] Locate datasets automatically
def label_from_folder(name):
    s = name.strip().lower().replace("_", " ")
    if "negative" in s or s == "nilm":
        return "NILM"
    if s.startswith("low") or s == "lsil":
        return "LSIL"
    if s.startswith("high") or s == "hsil":
        return "HSIL"
    if "carcinoma" in s or s in ("scc", "ssc"):
        return "SCC"
    return None


def subdir_labels(d):
    try:
        subs = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
    except OSError:
        return {}
    return {x: label_from_folder(x) for x in subs if label_from_folder(x)}


def find_roots():
    lbc = mal = sip = None
    known = {
        "lbc": ["/kaggle/input/datasets/bhavanamohod16/lbc-dataset1", "/kaggle/input/lbc-dataset1"],
        "mal": ["/kaggle/input/datasets/bhavanamohod16/malhari/Malhari Dataset", "/kaggle/input/malhari/Malhari Dataset"],
        "sip": ["/kaggle/input/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed",
                "/kaggle/input/cervical-cancer-largest-dataset-sipakmed"],
    }
    for p in known["lbc"]:
        if len(set(subdir_labels(p).values())) == 4:
            lbc = p
    for p in known["mal"]:
        if len(set(subdir_labels(p).values())) == 4:
            mal = p
    for p in known["sip"]:
        if os.path.isdir(p) and any(x.lower() == "im_dyskeratotic" for x in os.listdir(p)):
            sip = p
    for root, dirs, files in os.walk(INPUT_ROOT):
        depth = root.count(os.sep) - INPUT_ROOT.count(os.sep)
        if depth > 6 or "revision_outputs" in root:
            dirs[:] = []
            continue
        labs = subdir_labels(root)
        if len(set(labs.values())) == 4:
            is_lbc = any("negative for intraepithelial" in k.lower() for k in labs)
            if "malhari" in root.lower() and mal is None:
                mal = root
            elif is_lbc and lbc is None:
                lbc = root
        if sip is None and any(x.lower() == "im_dyskeratotic" for x in dirs):
            sip = root
    return lbc, mal, sip


LBC_ROOT, MAL_ROOT, SIP_ROOT = find_roots()
print("LBC     :", LBC_ROOT); print("Malhari :", MAL_ROOT); print("SIPaKMeD:", SIP_ROOT)
assert LBC_ROOT and MAL_ROOT and SIP_ROOT, "A dataset was not found under /kaggle/input - attach LBC, Malhari and SIPaKMeD."

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def scan_bethesda(root, name):
    rows = []
    for sub, lab in subdir_labels(root).items():
        for dp, _, fs in os.walk(os.path.join(root, sub)):
            for f in sorted(fs):
                if f.lower().endswith(IMG_EXT):
                    rows.append((os.path.join(dp, f), lab, name))
    df = pd.DataFrame(rows, columns=["path", "label", "dataset"]).drop_duplicates("path")
    return df.sort_values("path").reset_index(drop=True)


def scan_sipakmed(root):
    rows = []
    for c in os.listdir(root):
        if c.lower() not in [s.lower() for s in SIP_CLASSES]:
            continue
        cname = [s for s in SIP_CLASSES if s.lower() == c.lower()][0]
        cdir = os.path.join(root, c)
        crops = [dp for dp, _, _ in os.walk(cdir) if os.path.basename(dp).upper() == "CROPPED"]
        for dp in crops:
            for f in sorted(os.listdir(dp)):
                if f.lower().endswith(IMG_EXT):
                    cluster = f.split("_")[0].split(".")[0]           # 001_03.bmp -> cluster image 001
                    rows.append((os.path.join(dp, f), cname, "SIPaKMeD", f"SIP:{cname}:{cluster}"))
        if not crops:                                                 # fallback: cluster images
            for dp, _, fs in os.walk(cdir):
                for f in sorted(fs):
                    if f.lower().endswith(IMG_EXT):
                        rows.append((os.path.join(dp, f), cname, "SIPaKMeD", f"SIP:{cname}:{f}"))
    return pd.DataFrame(rows, columns=["path", "label", "dataset", "group"]).sort_values("path").reset_index(drop=True)


DF_LBC = scan_bethesda(LBC_ROOT, "LBC")
DF_MAL = scan_bethesda(MAL_ROOT, "Malhari")
DF_SIP = scan_sipakmed(SIP_ROOT)
for n, d in [("LBC", DF_LBC), ("Malhari", DF_MAL), ("SIPaKMeD", DF_SIP)]:
    print(f"{n}: {len(d)} images", d.label.value_counts().to_dict())


# %% [4] Duplicate audit (MD5 + perceptual hash) and image groups
def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def phash(p):
    a = np.asarray(Image.open(p).convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float64)
    d = dctn(a, norm="ortho")[:8, :8].flatten()[1:]
    return (d > np.median(d)).astype(np.uint8)


class UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def duplicate_audit():
    fp = f"{OUT}/audit/groups.csv"
    if os.path.exists(fp):
        log("audit: loading previous groups")
        return pd.read_csv(fp)
    log("audit: hashing LBC + Malhari images")
    df = pd.concat([DF_LBC, DF_MAL], ignore_index=True)
    df["md5"] = [md5(p) for p in df.path]
    H = np.stack([phash(p) for p in df.path])
    pairs = []
    for _, g in df.groupby("md5"):
        ix = g.index.tolist()
        pairs += [(ix[0], j, "exact", 0) for j in ix[1:]]
    D = (H[:, None, :] != H[None, :, :]).sum(2)
    ii, jj = np.where(np.triu(D <= CFG["PHASH_HAMMING"], 1))
    pairs += [(i, j, "phash", int(D[i, j])) for i, j in zip(ii, jj) if (i, j) not in {(a, b) for a, b, _, _ in pairs}]
    pr = pd.DataFrame(pairs, columns=["i", "j", "type", "dist"])
    pr["path_i"], pr["path_j"] = df.path.values[pr.i], df.path.values[pr.j]
    pr["dataset_i"], pr["dataset_j"] = df.dataset.values[pr.i], df.dataset.values[pr.j]
    pr["label_i"], pr["label_j"] = df.label.values[pr.i], df.label.values[pr.j]
    pr.to_csv(f"{OUT}/audit/duplicate_pairs.csv", index=False)
    uf = UnionFind(range(len(df)))
    for i, j in zip(pr.i, pr.j):
        uf.union(i, j)
    df["group"] = [f"G{uf.find(i)}" for i in range(len(df))]
    summ = {
        "n_images": int(len(df)), "pairs_exact": int((pr.type == "exact").sum()),
        "pairs_phash": int((pr.type == "phash").sum()),
        "pairs_within_LBC": int(((pr.dataset_i == "LBC") & (pr.dataset_j == "LBC")).sum()),
        "pairs_within_Malhari": int(((pr.dataset_i == "Malhari") & (pr.dataset_j == "Malhari")).sum()),
        "pairs_cross_dataset": int((pr.dataset_i != pr.dataset_j).sum()),
        "pairs_label_conflict": int((pr.label_i != pr.label_j).sum()),
        "groups_LBC": int(df[df.dataset == "LBC"].group.nunique()),
        "groups_Malhari": int(df[df.dataset == "Malhari"].group.nunique()),
        "phash_hamming_threshold": CFG["PHASH_HAMMING"],
    }
    json.dump(summ, open(f"{OUT}/audit/audit_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    # montage of up to 12 flagged pairs for visual confirmation
    if len(pr):
        k = min(12, len(pr))
        fig, ax = plt.subplots(k, 2, figsize=(5, 2.4 * k))
        ax = np.atleast_2d(ax)
        for r, (_, row) in enumerate(pr.head(k).iterrows()):
            for c, p in enumerate([row.path_i, row.path_j]):
                ax[r, c].imshow(Image.open(p).convert("RGB").resize((200, 150)))
                ax[r, c].set_title(f"{row['dataset_' + 'ij'[c]]}/{row['label_' + 'ij'[c]]} {row.type}", fontsize=6)
                ax[r, c].axis("off")
        fig.tight_layout(); fig.savefig(f"{OUT}/audit/duplicate_pairs_montage.png", dpi=110); plt.close(fig)
    df[["path", "label", "dataset", "group"]].to_csv(fp, index=False)
    return df[["path", "label", "dataset", "group"]]


GROUPS = duplicate_audit()
GMAP = dict(zip(GROUPS.path, GROUPS.group))
DF_LBC["group"] = DF_LBC.path.map(GMAP)
DF_MAL["group"] = DF_MAL.path.map(GMAP)


# %% [5] Leakage-safe splits (saved once, reused by every session)
def sgkf_split(df, n_splits, seed, strat_col="strat"):
    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgk.split(np.zeros(len(df)), df[strat_col].values, df["group"].values))


def make_dev_test(df, test_frac, seed):
    df = df.copy().reset_index(drop=True)
    df["strat"] = df.label + "|" + df.dataset
    dev_i, test_i = sgkf_split(df, int(round(1 / test_frac)), seed)[0]
    df["partition"] = "dev"; df.loc[test_i, "partition"] = "test"
    df["fold"] = -1
    dev = df[df.partition == "dev"]
    for k, (_, va) in enumerate(sgkf_split(dev, CFG["N_FOLDS"], seed + 1)):
        df.loc[dev.index[va], "fold"] = k
    return df


def check_no_leak(df):
    key = df.partition.astype(str) + "/" + df.fold.astype(str)
    bad = df.assign(k=key).groupby("group").k.nunique()
    assert (bad <= 1).all(), f"group leakage: {list(bad[bad > 1].index[:5])}"


def get_splits():
    fp = f"{OUT}/splits/splits.json"
    if os.path.exists(fp):
        log("splits: loading saved splits")
        sp = json.load(open(fp))
        return {k: pd.DataFrame(v) for k, v in sp.items()}
    sp = {}
    lbc = make_dev_test(DF_LBC, CFG["TEST_FRAC"], 42); check_no_leak(lbc); sp["lbc"] = lbc
    # Malhari target domain: 25 % adapt (unlabelled for UDA; labelled for few-shot), 75 % target-test
    m = DF_MAL.copy().reset_index(drop=True); m["strat"] = m.label
    tt_i, ad_i = sgkf_split(m, int(round(1 / CFG["MAL_ADAPT_FRAC"])), 42)[0]
    m["partition"] = "target_test"; m.loc[ad_i, "partition"] = "adapt"; m["fold"] = -1
    ad = m[m.partition == "adapt"]
    fs_tr, fs_va = sgkf_split(ad, 5, 43)[0]                     # adapt -> 80 % fine-tune train / 20 % val
    m.loc[ad.index[fs_tr], "partition"] = "adapt_train"; m.loc[ad.index[fs_va], "partition"] = "adapt_val"
    check_no_leak(m); sp["mal_target"] = m
    # multi-domain: LBC uses the same test set; Malhari gets its own 20 % hold-out
    mm = make_dev_test(DF_MAL, CFG["MAL_MERGED_TEST_FRAC"], 44)
    lb = lbc.copy()
    merged = pd.concat([lb[lb.partition == "test"], lb[lb.partition == "dev"].assign(fold=-1),
                        mm[mm.partition == "test"], mm[mm.partition == "dev"].assign(fold=-1)], ignore_index=True)
    merged["strat"] = merged.label + "|" + merged.dataset
    dev = merged[merged.partition == "dev"]
    for k, (_, va) in enumerate(sgkf_split(dev, CFG["N_FOLDS"], 45)):
        merged.loc[dev.index[va], "fold"] = k
    check_no_leak(merged); sp["merged"] = merged
    # SIPaKMeD: grouped by parent cluster image, 70 / 10 / 20
    s = DF_SIP.copy().reset_index(drop=True); s["strat"] = s.label
    dev_i, te_i = sgkf_split(s, 5, 46)[0]
    s["partition"] = "dev"; s.loc[te_i, "partition"] = "test"; s["fold"] = -1
    dv = s[s.partition == "dev"]
    tr_i, va_i = sgkf_split(dv, 8, 47)[0]
    s.loc[dv.index[va_i], "partition"] = "val"
    check_no_leak(s); sp["sip"] = s
    json.dump({k: v.to_dict(orient="list") for k, v in sp.items()}, open(fp, "w"))
    for k, v in sp.items():
        v.to_csv(f"{OUT}/splits/{k}.csv", index=False)
    return sp


SPLITS = get_splits()
for k, v in SPLITS.items():
    key = np.where(v.fold >= 0, "fold" + v.fold.astype(str), v.partition)
    print(f"\n== split {k}\n", pd.crosstab(key, v.label, margins=True))


# %% [6] Image cache, transforms, datasets
_CACHE = {}


def cached(path):
    im = _CACHE.get(path)
    if im is None:
        im = np.asarray(Image.open(path).convert("RGB").resize((CFG["IMG"], CFG["IMG"]), Image.BILINEAR))
        _CACHE[path] = im
    return im


def warm_cache(paths):
    todo = [p for p in paths if p not in _CACHE]
    if todo:
        log(f"caching {len(todo)} images at {CFG['IMG']} px")
        for p in todo:
            cached(p)


MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TRAIN_TF = transforms.Compose([                     # identical augmentation to the original notebook
    transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.05)),
])
EVAL_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class ImgDS(Dataset):
    def __init__(self, df, classes, train):
        self.p = df.path.tolist()
        m = {c: i for i, c in enumerate(classes)}
        self.y = [m.get(l, -1) for l in df.label]
        self.tf = TRAIN_TF if train else EVAL_TF

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(cached(self.p[i]))), self.y[i]


def loader(df, classes, train, bs=None, shuffle=None):
    bs = bs or CFG["BS"]
    ds = ImgDS(df, classes, train)
    drop = train and len(ds) > bs
    return DataLoader(ds, batch_size=min(bs, len(ds)), shuffle=train if shuffle is None else shuffle,
                      drop_last=drop, num_workers=CFG["NUM_WORKERS"], pin_memory=CUDA,
                      persistent_workers=False)


# %% [7] Model - identical to the original DeepCervixModel
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool, self.max_pool = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu1(self.fc1(self.avg_pool(x)))) +
                            self.fc2(self.relu1(self.fc1(self.max_pool(x)))))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv1(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1)))


class ProperCBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca, self.sa = ChannelAttention(ch), SpatialAttention()

    def forward(self, x):
        out = x * self.ca(x)
        return out * self.sa(out)


def imagenet_weights(w):
    return None if SMOKE else w


class DeepCervixModel(nn.Module):
    """EfficientNet-B3 + Progressive CBAM (CBAM after the 48-, 136- and 384-channel stages and on the
    final 1536-channel map) + LayerNorm/projection head. use_cbam=False gives the no-attention ablation."""

    def __init__(self, num_classes, imagenet=True, use_cbam=True):
        super().__init__()
        base = efficientnet_b3(weights=imagenet_weights(EfficientNet_B3_Weights.DEFAULT) if imagenet else None)
        self.features = base.features
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam1, self.cbam2, self.cbam3 = ProperCBAM(48), ProperCBAM(136), ProperCBAM(384)
            self.cbam_final = ProperCBAM(1536)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ln = nn.LayerNorm(1536)
        self.projection = nn.Sequential(nn.Linear(1536, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
                                        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, return_feat=False):
        f = x
        for _, module in self.features.named_children():
            f = module(f)
            if self.use_cbam:
                if f.size(1) == 48:
                    f = self.cbam1(f)
                elif f.size(1) == 136:
                    f = self.cbam2(f)
                elif f.size(1) == 384:
                    f = self.cbam3(f)
        if self.use_cbam:
            f = self.cbam_final(f)
        pooled = self.ln(self.pool(f).flatten(1))
        feat = self.projection(pooled)
        logits = self.fc(feat)
        return (logits, feat) if return_feat else logits


class TVBaseline(nn.Module):
    """Same-protocol baselines with the same projection head."""

    def __init__(self, name, num_classes):
        super().__init__()
        if name == "resnet50":
            b = resnet50(weights=imagenet_weights(ResNet50_Weights.DEFAULT)); d = b.fc.in_features; b.fc = nn.Identity()
        elif name == "densenet121":
            b = densenet121(weights=imagenet_weights(DenseNet121_Weights.DEFAULT)); d = b.classifier.in_features; b.classifier = nn.Identity()
        else:
            raise ValueError(name)
        self.backbone = b
        self.ln = nn.LayerNorm(d)
        self.projection = nn.Sequential(nn.Linear(d, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
                                        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, return_feat=False):
        feat = self.projection(self.ln(self.backbone(x)))
        logits = self.fc(feat)
        return (logits, feat) if return_feat else logits


SIP_CKPT = f"{OUT}/sipakmed_pretrained.pth"


def build_model(arch, num_classes, init):
    if arch == "pcbam":
        m = DeepCervixModel(num_classes, imagenet=(init == "imagenet"))
    elif arch == "effb3_nocbam":
        m = DeepCervixModel(num_classes, imagenet=True, use_cbam=False)
    else:
        m = TVBaseline(arch, num_classes)
    if init == "sipakmed":
        if not os.path.exists(SIP_CKPT):
            raise FileNotFoundError(f"SIPaKMeD checkpoint missing: {SIP_CKPT}")   # fail loudly (bug in v1)
        sd = torch.load(SIP_CKPT, map_location="cpu")
        own = m.state_dict()
        use = {k: v for k, v in sd.items() if k in own and not k.startswith("fc.") and v.shape == own[k].shape}
        n_expected = len([k for k in own if not k.startswith("fc.")])
        assert len(use) == n_expected, f"only {len(use)}/{n_expected} SIPaKMeD tensors matched"
        own.update(use); m.load_state_dict(own)
        print(f"   SIPaKMeD weights loaded: {len(use)}/{n_expected} tensors")
    return m.to(DEVICE)


# %% [8] Losses (focal loss as in v1; CORAL + multi-kernel MMD)
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha.to(DEVICE) if alpha is not None else None
        self.gamma, self.ls = gamma, label_smoothing

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, weight=self.alpha, reduction="none", label_smoothing=self.ls)
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


def class_weights(labels, n):
    c = np.bincount(np.asarray(labels), minlength=n).astype(float)
    return torch.tensor(c.sum() / (n * np.maximum(c, 1)), dtype=torch.float32)


def coral_loss(fs, ft):
    d = fs.size(1)

    def cov(x):
        x = x - x.mean(0, keepdim=True)
        return x.t() @ x / max(x.size(0) - 1, 1)

    return ((cov(fs) - cov(ft)) ** 2).sum() / (4 * d * d)


def mk_mmd_loss(fs, ft, n_kernels=5):
    """Unbiased MK-MMD^2 on L2-normalised features; bandwidths = median heuristic x 2^{-2..2}."""
    fs, ft = F.normalize(fs, dim=1), F.normalize(ft, dim=1)
    x = torch.cat([fs, ft]); d2 = torch.cdist(x, x).pow(2)
    med = d2.detach()[d2.detach() > 0].median() if (d2 > 0).any() else torch.tensor(1.0, device=x.device)
    k = sum(torch.exp(-d2 / (med * 2.0 ** u)) for u in range(-2, 3)) / n_kernels
    ns, nt = fs.size(0), ft.size(0)
    kss, ktt, kst = k[:ns, :ns], k[ns:, ns:], k[:ns, ns:]
    return ((kss.sum() - kss.diag().sum()) / (ns * (ns - 1)) + (ktt.sum() - ktt.diag().sum()) / (nt * (nt - 1))
            - 2 * kst.mean())


# %% [9] Training engine (resumable, time-guarded)
def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=CUDA)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=CUDA)


def autocast():
    return torch.autocast(device_type="cuda", enabled=CUDA) if CUDA else torch.autocast(device_type="cpu", enabled=False)


@torch.no_grad()
def predict(model, df, classes):
    model.eval()
    zs, fs = [], []
    for x, _ in loader(df, classes, train=False, shuffle=False):
        with autocast():
            z, f = model(x.to(DEVICE), return_feat=True)
        zs.append(z.float().cpu()); fs.append(f.float().cpu())
    return torch.cat(zs).numpy(), torch.cat(fs).numpy()


def macro_f1_on(model, df, classes):
    z, _ = predict(model, df, classes)
    m = {c: i for i, c in enumerate(classes)}
    return f1_score([m[l] for l in df.label], z.argmax(1), average="macro")


def fit(model, train_df, val_df, classes, *, epochs, lr, patience, bs=None, target_df=None, select="val_f1"):
    """select='val_f1': keep best validation epoch (early stopping); select='last': fixed schedule."""
    m = {c: i for i, c in enumerate(classes)}
    crit = FocalLoss(class_weights([m[l] for l in train_df.label], len(classes)), CFG["FOCAL_GAMMA"], CFG["LABEL_SMOOTH"])
    tl = loader(train_df, classes, train=True, bs=bs)
    tg = None
    if target_df is not None:
        tg_loader = loader(target_df, classes, train=True, bs=bs)
        tg = itertools.cycle(tg_loader)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CFG["WD"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(tl), epochs=epochs)
    scaler = make_scaler()
    if CUDA:
        torch.cuda.reset_peak_memory_stats()
    best, best_state, bad, hist, step = -1.0, None, 0, [], 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        run = defaultdict(float)
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with autocast():
                z, fsrc = model(x, return_feat=True)
                loss = crit(z, y)
            run["ce"] += loss.item()
            if tg is not None:
                xt, _ = next(tg)
                with autocast():
                    _, ftg = model(xt.to(DEVICE), return_feat=True)
                p = step / max(epochs * len(tl), 1)
                lam = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
                lc = coral_loss(fsrc.float(), ftg.float()); lm = mk_mmd_loss(fsrc.float(), ftg.float())
                loss = loss + lam * (CFG["W_CORAL"] * lc + CFG["W_MMD"] * lm)
                run["coral"] += lc.item(); run["mmd"] += lm.item(); run["lambda"] = lam
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
        vf1 = macro_f1_on(model, val_df, classes)
        rec = {"epoch": ep + 1, "val_f1": float(vf1), **{k: (v / len(tl) if k != "lambda" else v) for k, v in run.items()}}
        hist.append(rec)
        if select == "last" or vf1 > best:
            best, bad = max(best, vf1), 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    info = {"history": hist, "train_time_s": time.time() - t0, "epochs_run": len(hist), "best_val_f1": best,
            "peak_train_mem_MB": torch.cuda.max_memory_allocated() / 2**20 if CUDA else float("nan")}
    return model, info


def run_dir(exp, seed, fold):
    return f"{OUT}/runs/{exp}/s{seed}/f{fold}"


def is_done(exp, seed, fold):
    return os.path.exists(f"{run_dir(exp, seed, fold)}/preds.npz")


def save_run(exp, seed, fold, model, info, evalsets, classes, keep_weights=False):
    d = run_dir(exp, seed, fold); os.makedirs(d, exist_ok=True)
    out = {}
    for name, df in evalsets.items():
        if df is None or len(df) == 0:
            continue
        z, f = predict(model, df, classes)
        m = {c: i for i, c in enumerate(classes)}
        out[f"{name}__logits"] = z; out[f"{name}__feats"] = f
        out[f"{name}__y"] = np.array([m.get(l, -1) for l in df.label])
        out[f"{name}__path"] = df.path.values.astype(str); out[f"{name}__dataset"] = df.dataset.values.astype(str)
    np.savez_compressed(f"{d}/preds.npz", **out)
    json.dump({**info, "classes": classes}, open(f"{d}/log.json", "w"), indent=1, default=float)
    if keep_weights:
        torch.save(model.state_dict(), f"{d}/model.pth")


DURATIONS = defaultdict(list)


def time_left_ok(kind):
    est = np.median(DURATIONS[kind]) if DURATIONS[kind] else 20 * 60
    return (time.time() - T_START) + est < CFG["MAX_HOURS"] * 3600


# %% [10] Experiment definitions
SPL_LBC, SPL_MAL, SPL_MER, SPL_SIP = SPLITS["lbc"], SPLITS["mal_target"], SPLITS["merged"], SPLITS["sip"]
LBC_TEST = SPL_LBC[SPL_LBC.partition == "test"]
MAL_TT = SPL_MAL[SPL_MAL.partition == "target_test"]
MAL_ADAPT = SPL_MAL[SPL_MAL.partition.isin(["adapt_train", "adapt_val"])]
MAL_FS_TR, MAL_FS_VA = SPL_MAL[SPL_MAL.partition == "adapt_train"], SPL_MAL[SPL_MAL.partition == "adapt_val"]


def lbc_fold(fold):
    dev = SPL_LBC[SPL_LBC.partition == "dev"]
    return dev[dev.fold != fold], dev[dev.fold == fold]


def task_sip_pretrain():
    if os.path.exists(SIP_CKPT) and os.path.exists(f"{run_dir('sip_pre', 0, 0)}/preds.npz"):
        return "skip"
    seed_all(0)
    tr, va, te = [SPL_SIP[SPL_SIP.partition == p] for p in ["dev", "val", "test"]]
    warm_cache(SPL_SIP.path)
    m = build_model("pcbam", len(SIP_CLASSES), "imagenet")
    m, info = fit(m, tr, va, SIP_CLASSES, epochs=CFG["SIP_EPOCHS"], lr=CFG["SIP_LR"], patience=CFG["PATIENCE"])
    torch.save(m.state_dict(), SIP_CKPT)
    save_run("sip_pre", 0, 0, m, info, {"val": va, "sip_test": te}, SIP_CLASSES)
    return info


def task_lbc(init, seed, fold):
    exp = f"lbc_{'in' if init == 'imagenet' else 'sip'}"
    if is_done(exp, seed, fold) and os.path.exists(f"{run_dir(exp, seed, fold)}/model.pth"):
        return "skip"
    seed_all(seed * 1000 + fold)
    tr, va = lbc_fold(fold)
    m = build_model("pcbam", 4, init)
    m, info = fit(m, tr, va, BETHESDA, epochs=CFG["MAX_EPOCHS"], lr=CFG["LR"], patience=CFG["PATIENCE"])
    save_run(exp, seed, fold, m, info, {"val": va, "lbc_test": LBC_TEST, "mal_tt": MAL_TT}, BETHESDA, keep_weights=True)
    return info


def load_base(init, seed, fold):
    exp = f"lbc_{'in' if init == 'imagenet' else 'sip'}"
    p = f"{run_dir(exp, seed, fold)}/model.pth"
    if not os.path.exists(p):
        task_lbc(init, seed, fold)
    m = build_model("pcbam", 4, "imagenet" if init == "imagenet" else "scratch")
    m.load_state_dict(torch.load(p, map_location=DEVICE))
    return m


def task_uda(init, seed, fold):
    exp = f"lbc_{'in' if init == 'imagenet' else 'sip'}_da"
    if is_done(exp, seed, fold):
        return "skip"
    seed_all(seed * 1000 + fold + 7)
    tr, va = lbc_fold(fold)
    m = load_base(init, seed, fold)
    # unlabelled target images (labels never used): Malhari adapt partition only
    m, info = fit(m, tr, va, BETHESDA, epochs=CFG["UDA_EPOCHS"], lr=CFG["UDA_LR"], patience=999,
                  target_df=MAL_ADAPT, select="last")
    save_run(exp, seed, fold, m, info, {"val": va, "lbc_test": LBC_TEST, "mal_tt": MAL_TT}, BETHESDA)
    return info


def task_fewshot(init, seed, fold):
    exp = f"lbc_{'in' if init == 'imagenet' else 'sip'}_fs"
    if is_done(exp, seed, fold):
        return "skip"
    seed_all(seed * 1000 + fold + 13)
    m = load_base(init, seed, fold)
    # labelled Malhari adapt images: train part for gradient steps, val part for early stopping
    m, info = fit(m, MAL_FS_TR, MAL_FS_VA, BETHESDA, epochs=CFG["FS_EPOCHS"], lr=CFG["FS_LR"],
                  patience=CFG["FS_PATIENCE"], bs=CFG["FS_BS"])
    save_run(exp, seed, fold, m, info, {"val": MAL_FS_VA, "lbc_test": LBC_TEST, "mal_tt": MAL_TT}, BETHESDA)
    return info


def task_merged(init, seed, fold):
    exp = f"merged_{'in' if init == 'imagenet' else 'sip'}"
    if is_done(exp, seed, fold):
        return "skip"
    seed_all(seed * 1000 + fold + 21)
    dev = SPL_MER[SPL_MER.partition == "dev"]
    tr, va = dev[dev.fold != fold], dev[dev.fold == fold]
    te = SPL_MER[SPL_MER.partition == "test"]
    m = build_model("pcbam", 4, init)
    m, info = fit(m, tr, va, BETHESDA, epochs=CFG["MAX_EPOCHS"], lr=CFG["LR"], patience=CFG["PATIENCE"])
    save_run(exp, seed, fold, m, info, {"val": va, "merged_test": te}, BETHESDA)
    return info


def task_baseline(arch, seed, fold):
    exp = f"base_{arch}"
    if is_done(exp, seed, fold):
        return "skip"
    seed_all(seed * 1000 + fold + 31)
    tr, va = lbc_fold(fold)
    m = build_model(arch, 4, "imagenet")
    m, info = fit(m, tr, va, BETHESDA, epochs=CFG["MAX_EPOCHS"], lr=CFG["LR"], patience=CFG["PATIENCE"])
    save_run(exp, seed, fold, m, info, {"val": va, "lbc_test": LBC_TEST, "mal_tt": MAL_TT}, BETHESDA)
    return info


CASCADE_TASKS = {"binary": (["NILM", "Abnormal"], {"NILM": "NILM", "LSIL": "Abnormal", "HSIL": "Abnormal", "SCC": "Abnormal"}),
                 "severity": (["LSIL", "HSIL", "SCC"], {"LSIL": "LSIL", "HSIL": "HSIL", "SCC": "SCC"})}


def task_cascade(stage, seed, fold):
    exp = f"cascade_{stage}"
    if is_done(exp, seed, fold):
        return "skip"
    seed_all(seed * 1000 + fold + 41)
    classes, mp = CASCADE_TASKS[stage]
    tr, va = lbc_fold(fold)
    tr = tr[tr.label.isin(mp)].assign(label=lambda d: d.label.map(mp))
    va = va[va.label.isin(mp)].assign(label=lambda d: d.label.map(mp))
    m = build_model("pcbam", len(classes), "imagenet")
    m, info = fit(m, tr, va, classes, epochs=CFG["MAX_EPOCHS"], lr=CFG["LR"], patience=CFG["PATIENCE"])
    # predict on ALL LBC test images (labels stay Bethesda for end-to-end evaluation)
    save_run(exp, seed, fold, m, info, {"val": va, "lbc_test": LBC_TEST}, classes)
    return info


def build_queue():
    q = [("sip_pretrain", task_sip_pretrain, ())]
    folds = range(CFG["N_FOLDS"])

    def main_seed(s):
        for f in folds:
            for init in ["imagenet", "sipakmed"]:
                q.append((f"lbc_{init}", task_lbc, (init, s, f)))
                q.append((f"uda_{init}", task_uda, (init, s, f)))
                q.append((f"fewshot_{init}", task_fewshot, (init, s, f)))

    main_seed(CFG["SEEDS_MAIN"][0])
    for s in CFG["SEEDS_OTHER"]:
        for f in folds:
            for init in ["imagenet", "sipakmed"]:
                q.append((f"merged_{init}", task_merged, (init, s, f)))
    if CFG["RUN_CASCADE"]:
        for s in CFG["SEEDS_OTHER"]:
            for f in folds:
                for st in ["binary", "severity"]:
                    q.append((f"cascade_{st}", task_cascade, (st, s, f)))
    for s in CFG["SEEDS_MAIN"][1:]:
        main_seed(s)
    if CFG["RUN_BASELINES"]:
        for a in CFG["BASELINES"]:
            for s in CFG["SEEDS_OTHER"]:
                for f in folds:
                    q.append((f"base_{a}", task_baseline, (a, s, f)))
    return q


# %% [11] Run the experiment queue (skips finished work; stops before the time limit)
def run_queue():
    warm_cache(pd.concat([SPL_LBC.path, SPL_MAL.path]).unique())
    q = build_queue()
    done = 0
    for i, (kind, fn, args) in enumerate(q):
        if not time_left_ok(kind):
            log(f"time budget reached - {len(q) - i} tasks left for the next session")
            return False
        t0 = time.time()
        r = fn(*args)
        if r == "skip":
            done += 1
            continue
        DURATIONS[kind].append(time.time() - t0)
        done += 1
        log(f"[{done}/{len(q)}] {kind}{args} done in {(time.time() - t0) / 60:.1f} min "
            f"(epochs {r.get('epochs_run')}, best val F1 {r.get('best_val_f1', float('nan')):.4f})")
    log("all tasks finished")
    return True


ALL_DONE = run_queue()
json.dump({"all_tasks_done": ALL_DONE, "hours": (time.time() - T_START) / 3600}, open(f"{OUT}/status.json", "w"))

# %% [12] Evaluation helpers: metrics, calibration, ensembles, statistics
def ece_score(y, p, n_bins=15):
    conf, acc = p.max(1), (p.argmax(1) == y).astype(float)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(acc[s].mean() - conf[s].mean())
    return e


def entropy(p, norm=False):
    h = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    return h / np.log(p.shape[1]) if norm else h


def metrics(y, p, classes):
    k = p.shape[1]; yh = p.argmax(1)
    r = {"accuracy": accuracy_score(y, yh), "balanced_accuracy": balanced_accuracy_score(y, yh),
         "precision": precision_score(y, yh, average="macro", zero_division=0),
         "recall": recall_score(y, yh, average="macro", zero_division=0),
         "f1": f1_score(y, yh, average="macro", zero_division=0)}
    try:
        r["auc"] = roc_auc_score(y, p[:, 1]) if k == 2 else roc_auc_score(y, p, multi_class="ovr", labels=list(range(k)))
    except ValueError:
        r["auc"] = float("nan")
    rec = recall_score(y, yh, average=None, labels=list(range(k)), zero_division=0)
    pre = precision_score(y, yh, average=None, labels=list(range(k)), zero_division=0)
    for i, c in enumerate(classes):
        r[f"recall_{c}"], r[f"precision_{c}"], r[f"support_{c}"] = rec[i], pre[i], int((y == i).sum())
    onehot = np.eye(k)[y]
    r["ece"] = ece_score(y, p); r["brier"] = float(((p - onehot) ** 2).sum(1).mean())
    r["nll"] = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
    order = np.argsort(entropy(p)); err = (yh != y).astype(float)[order]
    r["aurc"] = float((np.cumsum(err) / np.arange(1, len(err) + 1)).mean())
    e = (yh != y).astype(int)
    r["err_auroc"] = float(roc_auc_score(e, entropy(p))) if 0 < e.sum() < len(e) else float("nan")
    keep = order[: int(round(0.9 * len(y)))]
    r["error_rate_all"] = float((yh != y).mean()); r["error_rate_keep90"] = float((yh[keep] != y[keep]).mean())
    r["n"] = int(len(y)); r["cm"] = confusion_matrix(y, yh, labels=list(range(k))).tolist()
    return r


def fit_temperature(z, y):
    zt, yt = torch.tensor(z, dtype=torch.float64), torch.tensor(y, dtype=torch.long)
    lt = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([lt], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad(); l = F.cross_entropy(zt / lt.exp(), yt); l.backward(); return l

    opt.step(closure)
    return float(lt.exp().item())


RULES = ["uniform_prob", "uniform_logit", "entropy_weight", "v1_w_prop_uncertainty"]


def aggregate(Zs, rule, temps=None, tau=CFG["TAU"]):
    temps = temps or [1.0] * len(Zs)
    Z = np.stack([z / t for z, t in zip(Zs, temps)]); P = softmax(Z, axis=2)
    if rule == "uniform_prob":
        return P.mean(0)
    if rule == "uniform_logit":
        return softmax(Z.mean(0), axis=1)
    H = np.stack([entropy(p, True) for p in P])
    W = softmax(-H / tau, axis=0) if rule == "entropy_weight" else (H + 1e-8) / (H + 1e-8).sum(0, keepdims=True)
    return (W[..., None] * P).sum(0)


def load_seed(exp, seed):
    fs = [f"{run_dir(exp, seed, f)}/preds.npz" for f in range(CFG["N_FOLDS"])]
    if not all(os.path.exists(f) for f in fs):
        return None
    return [dict(np.load(f, allow_pickle=True)) for f in fs]


def seeds_of(exp):
    return sorted(int(os.path.basename(p)[1:]) for p in glob.glob(f"{OUT}/runs/{exp}/s*") if load_seed(exp, int(os.path.basename(p)[1:])))


def ens_eval(exp, seed, evalset, classes=BETHESDA, subset=None):
    R = load_seed(exp, seed)
    if R is None or f"{evalset}__logits" not in R[0]:
        return None
    y = R[0][f"{evalset}__y"]; sel = np.ones(len(y), bool) if subset is None else (R[0][f"{evalset}__dataset"] == subset)
    y = y[sel]; Zs = [r[f"{evalset}__logits"][sel] for r in R]
    temps = [fit_temperature(r["val__logits"], r["val__y"]) for r in R]
    out = {"single": [metrics(y, softmax(z, 1), classes) for z in Zs], "temps": temps, "y": y, "probs": {}}
    for ts in (False, True):
        for rule in RULES:
            name = rule + ("+TS" if ts else ""); p = aggregate(Zs, rule, temps if ts else None)
            out["probs"][name] = p; out[name] = metrics(y, p, classes)
    out["tau_sweep"] = {t: metrics(y, aggregate(Zs, "entropy_weight", temps, t), classes)["f1"] for t in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 1e6]}
    out["paths"] = R[0][f"{evalset}__path"][sel]
    return out


def boot_ci(y, p, fn, n=None, seed=0):
    n = n or CFG["N_BOOT"]; rng = np.random.default_rng(seed); vals = []
    idx_c = [np.where(y == c)[0] for c in np.unique(y)]
    for _ in range(n):
        ix = np.concatenate([rng.choice(i, len(i)) for i in idx_c])
        try:
            vals.append(fn(y[ix], p[ix]))
        except ValueError:
            pass
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def mcnemar_exact(y, a, b):
    n01, n10 = int(((a == y) & (b != y)).sum()), int(((a != y) & (b == y)).sum())
    return 1.0 if n01 + n10 == 0 else float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)


def holm_max(ps):
    ps = sorted(ps); m = len(ps); adj, run = [], 0
    for i, p in enumerate(ps):
        run = max(run, min(1, (m - i) * p)); adj.append(run)
    return max(adj) if adj else float("nan")


def paired_tests(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    out = {"n": len(a), "mean_a": a.mean(), "mean_b": b.mean(), "diff": d.mean()}
    out["t_p"] = float(stats.ttest_rel(a, b).pvalue) if len(a) > 1 and d.std() > 0 else float("nan")
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        out["wilcoxon_p"] = float("nan")
    return out


def ms(vals, pct=False, nd=4):
    v = np.asarray([x for x in vals if x == x], float)
    if len(v) == 0:
        return "--"
    f = 100 if pct else 1; nd = 2 if pct else nd
    return f"{f * v.mean():.{nd}f} $\\pm$ {f * v.std(ddof=1):.{nd}f}" if len(v) > 1 else f"{f * v.mean():.{nd}f}"


# %% [13] Build every table, figure and number for the manuscript
MAIN = "entropy_weight+TS"
RES = {}            # (exp, evalset, subset) -> list of per-seed ens_eval
VALUES = {}         # flat key -> value, for filling the manuscript placeholders


def get(exp, evalset, subset=None):
    key = (exp, evalset, subset)
    if key not in RES:
        RES[key] = [r for r in (ens_eval(exp, s, evalset, subset=subset) for s in seeds_of(exp)) if r]
    return RES[key]


def per_model(exp, evalset, metric, subset=None):
    return [m[metric] for r in get(exp, evalset, subset) for m in r["single"]]


def per_seed(exp, evalset, metric, rule=MAIN, subset=None):
    return [r[rule][metric] for r in get(exp, evalset, subset)]


def write_tex(name, body):
    open(f"{OUT}/generated/{name}.tex", "w").write(body + "\n")
    print(f"   wrote generated/{name}.tex")


# --- Table 1: splits
rows = []
for lab, df in [("Mendeley LBC", SPL_LBC), ("Merged LBC+Malhari", SPL_MER)]:
    n, te = len(df), (df.partition == "test").sum(); fold = df[df.partition == "dev"].groupby("fold").size()
    rows.append(f"{lab} & {n} & {df.group.nunique()} & {n - te} ({100 * (n - te) / n:.0f}\\%) & "
                f"{fold.mean():.0f} ({100 * fold.mean() / n:.0f}\\%) & {te} ({100 * te / n:.0f}\\%) \\\\")
n = len(SPL_MAL); ad = SPL_MAL.partition.isin(["adapt_train", "adapt_val"]).sum()
rows.append(f"Malhari (target) & {n} & {SPL_MAL.group.nunique()} & adapt {ad} ({100 * ad / n:.0f}\\%; "
            f"{(SPL_MAL.partition == 'adapt_train').sum()} train / {(SPL_MAL.partition == 'adapt_val').sum()} val) & -- & "
            f"{n - ad} ({100 * (n - ad) / n:.0f}\\%) \\\\")
s = SPL_SIP; n = len(s)
rows.append(f"SIPaKMeD cells & {n} & {s.group.nunique()} clusters & train {(s.partition == 'dev').sum()} / val "
            f"{(s.partition == 'val').sum()} & -- & {(s.partition == 'test').sum()} ({100 * (s.partition == 'test').mean():.0f}\\%) \\\\")
write_tex("tab_splits", "\n".join(rows))
VALUES["audit"] = json.load(open(f"{OUT}/audit/audit_summary.json"))
VALUES["n_LBC_test"] = int(len(LBC_TEST)); VALUES["n_Malhari_target_test"] = int(len(MAL_TT))
VALUES["n_Malhari_adapt"] = int(len(MAL_ADAPT)); VALUES["n_SCC_LBC_test"] = int((LBC_TEST.label == "SCC").sum())
VALUES["n_Malhari_merged_test"] = int(((SPL_MER.partition == "test") & (SPL_MER.dataset == "Malhari")).sum())

# --- Table 7 (main)
MAIN_ROWS = [
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
rows, csv_rows = [], []
for lab, exp, es, sub, pre, tr, te in MAIN_ROWS:
    R = get(exp, es, sub)
    if not R:
        continue
    v = {k: per_seed(exp, es, k, subset=sub) for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"]}
    rows.append(f"{lab} & {pre} & {tr} & {te} & {ms(v['accuracy'], True)} & {ms(v['balanced_accuracy'], True)} & "
                f"{ms(v['f1'])} & {ms(v['auc'])} \\\\")
    csv_rows.append({"model": lab.replace("\\;", ""), "init": pre, "train": tr, "test": te, "seeds": len(R),
                     **{f"{k}_mean": np.mean(x) for k, x in v.items()}, **{f"{k}_sd": np.std(x, ddof=1) if len(x) > 1 else 0 for k, x in v.items()},
                     "single_model_acc_mean": np.mean(per_model(exp, es, "accuracy", sub))})
    VALUES[f"acc_{exp}_{es}_{sub}"] = 100 * np.mean(v["accuracy"]); VALUES[f"f1_{exp}_{es}_{sub}"] = np.mean(v["f1"])
    VALUES[f"auc_{exp}_{es}_{sub}"] = np.mean(v["auc"])
if rows:
    write_tex("tab_main", "\n".join(rows))
    pd.DataFrame(csv_rows).to_csv(f"{OUT}/results/table_main.csv", index=False)

# --- Table 9: aggregation rules, calibration, uncertainty
blocks = []
names = [("Single model (mean of 5 folds)", "single"), ("Uniform probability average", "uniform_prob"),
         ("Uniform logit average", "uniform_logit"), ("Entropy-weighted (Eq.~\\ref{eq:w})", "entropy_weight"),
         ("Entropy-weighted + TS", "entropy_weight+TS"), ("$w\\propto$ uncertainty (v1 typo, ablation)", "v1_w_prop_uncertainty")]
calib_csv = []
for exp, es, sub, title in [("lbc_in", "lbc_test", None, "LBC test (in-domain)"), ("lbc_in", "mal_tt", None, "Malhari 75\\% (cross-domain)"),
                            ("merged_in", "merged_test", None, "Merged test (multi-domain)")]:
    R = get(exp, es, sub)
    if not R:
        continue
    rr = [f"\\multicolumn{{8}}{{l}}{{\\textit{{{title}}}}} \\\\"]
    for lab, k in names:
        if k == "single":
            g = lambda m: [np.mean([s[m] for s in r["single"]]) for r in R]
        else:
            g = lambda m, k=k: [r[k][m] for r in R]
        rr.append(f"{lab} & {ms(g('accuracy'), True)} & {ms(g('f1'))} & {ms(g('ece'))} & {ms(g('brier'))} & "
                  f"{ms(g('nll'))} & {ms(g('aurc'))} & {ms(g('err_auroc'))} \\\\")
        calib_csv.append({"set": title, "rule": k, **{m: np.mean(g(m)) for m in ["accuracy", "f1", "ece", "brier", "nll", "aurc", "err_auroc"]}})
    blocks.append("\n".join(rr))
    VALUES[f"tau_sweep_{exp}_{es}"] = {str(t): float(np.mean([r["tau_sweep"][t] for r in R])) for t in R[0]["tau_sweep"]}
    VALUES[f"errAUROC_{exp}_{es}"] = float(np.mean([r[MAIN]["err_auroc"] for r in R]))
    VALUES[f"error_all_vs_keep90_{exp}_{es}"] = (float(np.mean([r[MAIN]["error_rate_all"] for r in R])),
                                                float(np.mean([r[MAIN]["error_rate_keep90"] for r in R])))
    VALUES[f"ece_uniform_vs_entropyTS_{exp}_{es}"] = (float(np.mean([r["uniform_prob"]["ece"] for r in R])),
                                                     float(np.mean([r[MAIN]["ece"] for r in R])))
if blocks:
    write_tex("tab_calibration", "\n\\midrule\n".join(blocks))
    pd.DataFrame(calib_csv).to_csv(f"{OUT}/results/table_calibration.csv", index=False)

# --- Table 8: statistical comparisons
cmp_rows, cmp_csv = [], []


def compare(label, a, b, es, level="model", rule_a=MAIN, rule_b=MAIN):
    Ra, Rb = get(a, es), get(b, es)
    n = min(len(Ra), len(Rb))
    if n == 0:
        return
    if level == "model":
        res = {m: paired_tests(per_model(a, es, m)[:5 * n], per_model(b, es, m)[:5 * n]) for m in ["accuracy", "f1", "auc"]}
    else:
        res = {m: paired_tests([r[rule_a][m] for r in Ra[:n]], [r[rule_b][m] for r in Rb[:n]]) for m in ["accuracy", "f1", "auc"]}
    mc = [mcnemar_exact(Ra[i]["y"], Ra[i]["probs"][rule_a].argmax(1), Rb[i]["probs"][rule_b].argmax(1)) for i in range(n)]
    y, pa, pb = Ra[0]["y"], Ra[0]["probs"][rule_a], Rb[0]["probs"][rule_b]
    lo, hi = boot_ci(y, np.stack([pa, pb], 1), lambda t, q: f1_score(t, q[:, 0].argmax(1), average="macro") - f1_score(t, q[:, 1].argmax(1), average="macro"))
    unit = "models" if level == "model" else "seeds"
    cmp_rows.append(f"{label} & {res['f1']['n']} {unit} & {100 * res['accuracy']['diff']:+.2f} ({res['accuracy']['t_p']:.3f}) & "
                    f"{res['f1']['diff']:+.4f} ({res['f1']['t_p']:.3f}) & {res['auc']['diff']:+.4f} ({res['auc']['t_p']:.3f}) & "
                    f"{res['f1']['wilcoxon_p']:.3f} & {holm_max(mc):.3f} \\\\")
    rec = {"comparison": label, "eval": es, "level": level, "mcnemar_p_per_seed": mc, "mcnemar_holm_max": holm_max(mc),
           "f1_diff_seed0_boot95": (lo, hi), **{f"{m}_{k}": v for m, d in res.items() for k, v in d.items()}}
    cmp_csv.append(rec); VALUES[f"cmp::{label}"] = rec


compare("ImageNet: CORAL+MK-MMD vs base (Malhari)", "lbc_in_da", "lbc_in", "mal_tt")
compare("SIPaKMeD: CORAL+MK-MMD vs base (Malhari)", "lbc_sip_da", "lbc_sip", "mal_tt")
compare("ImageNet: few-shot vs base (Malhari)", "lbc_in_fs", "lbc_in", "mal_tt")
compare("SIPaKMeD: few-shot vs base (Malhari)", "lbc_sip_fs", "lbc_sip", "mal_tt")
compare("SIPaKMeD vs ImageNet init (LBC test)", "lbc_sip", "lbc_in", "lbc_test")
compare("SIPaKMeD vs ImageNet init (Malhari)", "lbc_sip", "lbc_in", "mal_tt")
compare("Entropy-weighted+TS vs uniform (LBC)", "lbc_in", "lbc_in", "lbc_test", "ensemble", MAIN, "uniform_prob")
compare("Entropy-weighted+TS vs uniform (Malhari)", "lbc_in", "lbc_in", "mal_tt", "ensemble", MAIN, "uniform_prob")
if cmp_rows:
    write_tex("tab_da", "\n".join(cmp_rows))
    pd.DataFrame(cmp_csv).to_csv(f"{OUT}/results/table_statistics.csv", index=False)

# --- Table 10: same-protocol baselines and attention ablation
rows = []
for lab, exp in [("ResNet-50", "base_resnet50"), ("DenseNet-121", "base_densenet121"),
                 ("EfficientNet-B3 (no attention)", "base_effb3_nocbam"), ("EfficientNet-B3 + PCBAM (proposed)", "lbc_in")]:
    for es, te in [("lbc_test", "LBC"), ("mal_tt", "Malhari 75\\%")]:
        R = get(exp, es)
        if R:
            rows.append(f"{lab} & {te} & {ms(per_seed(exp, es, 'accuracy'), True)} & {ms(per_seed(exp, es, 'f1'))} & "
                        f"{ms(per_seed(exp, es, 'auc'))} & {ms(per_seed(exp, es, 'ece'))} \\\\")
if rows:
    write_tex("tab_baselines", "\n".join(rows))

# --- Table 5: SIPaKMeD pre-training task on held-out cluster-grouped test cells
R = ens_eval("sip_pre", 0, "sip_test", SIP_CLASSES) if os.path.exists(f"{run_dir('sip_pre', 0, 0)}/preds.npz") else None
if os.path.exists(f"{run_dir('sip_pre', 0, 0)}/preds.npz"):
    d = dict(np.load(f"{run_dir('sip_pre', 0, 0)}/preds.npz", allow_pickle=True))
    y = d["sip_test__y"]; p = softmax(d["sip_test__logits"], 1); m = metrics(y, p, SIP_CLASSES)
    rows = [f"{c.replace('im_', '')} & {m['precision_' + c]:.3f} & {m['recall_' + c]:.3f} & {m['support_' + c]} \\\\" for c in SIP_CLASSES]
    lo, hi = boot_ci(y, p, lambda t, q: accuracy_score(t, q.argmax(1)))
    rows.append(f"\\midrule Overall (single model, held-out cells) & \\multicolumn{{3}}{{l}}{{Acc {100 * m['accuracy']:.2f}\\% "
                f"[{100 * lo:.2f}, {100 * hi:.2f}], macro-F1 {m['f1']:.4f}}} \\\\")
    write_tex("tab_sipakmed", "\n".join(rows)); VALUES["sipakmed_test"] = {k: m[k] for k in ["accuracy", "f1", "auc"]}

# --- Table 6: two-stage cascade vs flat
def cascade_eval():
    b, s = load_seed("cascade_binary", 0), load_seed("cascade_severity", 0)
    if b is None or s is None:
        return None
    tb = [fit_temperature(r["val__logits"], r["val__y"]) for r in b]
    ts = [fit_temperature(r["val__logits"], r["val__y"]) for r in s]
    pb = aggregate([r["lbc_test__logits"] for r in b], "entropy_weight", tb)
    psv = aggregate([r["lbc_test__logits"] for r in s], "entropy_weight", ts)
    paths = b[0]["lbc_test__path"]; lab = LBC_TEST.set_index("path").loc[paths, "label"].values
    y = np.array([BETHESDA.index(l) for l in lab])
    abn = pb.argmax(1) == 1
    final = np.zeros((len(y), 4)); final[~abn, 0] = 1.0; final[abn, 1:] = psv[abn]
    res = {"cascade": metrics(y, final, BETHESDA),
           "stage1_sens": float((abn & (y > 0)).sum() / max((y > 0).sum(), 1)),
           "stage1_spec": float((~abn & (y == 0)).sum() / max((y == 0).sum(), 1)),
           "scc_missed": int(((y == 3) & ~abn).sum()), "n_scc": int((y == 3).sum())}
    lo, hi = boot_ci(y, final, lambda t, q: recall_score(t, q.argmax(1), labels=[3], average=None, zero_division=0)[0])
    res["scc_recall_ci"] = (lo, hi)
    fl = ens_eval("lbc_in", 0, "lbc_test")
    if fl:
        res["flat"] = fl[MAIN]
    return res


CAS = cascade_eval() if CFG["RUN_CASCADE"] else None
if CAS:
    rows = []
    for lab, k in [("Two-stage cascade", "cascade"), ("Flat 4-class ensemble", "flat")]:
        if k in CAS:
            d = CAS[k]
            rows.append(f"{lab} & {100 * d['accuracy']:.2f} & {d['f1']:.4f} & {d['recall_NILM']:.3f} & {d['recall_LSIL']:.3f} & "
                        f"{d['recall_HSIL']:.3f} & {d['recall_SCC']:.3f} \\\\")
    rows.append(f"\\multicolumn{{7}}{{l}}{{\\footnotesize Stage-1 sensitivity (abnormal) {CAS['stage1_sens']:.3f}, specificity "
                f"{CAS['stage1_spec']:.3f}; SCC missed by Stage 1: {CAS['scc_missed']} of {CAS['n_scc']}; cascade SCC recall 95\\% CI "
                f"[{CAS['scc_recall_ci'][0]:.2f}, {CAS['scc_recall_ci'][1]:.2f}].}} \\\\")
    write_tex("tab_cascade", "\n".join(rows))
    VALUES["cascade"] = {k: (v if not isinstance(v, dict) else {m: v[m] for m in ["accuracy", "f1", "recall_SCC"]}) for k, v in CAS.items()}


# %% [14] Computational cost of one member and of the five-member ensemble
def cost_profile():
    m = build_model("pcbam", 4, "scratch").eval()
    n_par = sum(p.numel() for p in m.parameters())
    x1 = torch.randn(1, 3, CFG["IMG"], CFG["IMG"], device=DEVICE)
    flops = float("nan")
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                m(x1)
        flops = fc.get_total_flops()
    except Exception as e:
        print("flop counter unavailable:", e)

    def lat(bs, reps=30 if not SMOKE else 2):
        x = torch.randn(bs, 3, CFG["IMG"], CFG["IMG"], device=DEVICE)
        if CUDA:
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(3):
                m(x)
            if CUDA:
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(reps):
                m(x)
            if CUDA:
                torch.cuda.synchronize()
        return (time.perf_counter() - t) / reps, (torch.cuda.max_memory_allocated() / 2**20 if CUDA else float("nan"))

    t1, _ = lat(1); t32, mem32 = lat(min(32, CFG["BS"] * 4))
    logs = [json.load(open(f)) for f in glob.glob(f"{OUT}/runs/lbc_in/s*/f*/log.json")]
    tt = [l["train_time_s"] for l in logs]
    return {"gpu": torch.cuda.get_device_name(0) if CUDA else "CPU", "params_M": n_par / 1e6, "GFLOPs": flops / 1e9,
            "GMACs": flops / 2e9, "size_MB": n_par * 4 / 2**20, "lat_ms_bs1": 1000 * t1, "thr_bs32": min(32, CFG["BS"] * 4) / t32,
            "mem_bs32_MB": mem32, "train_min_mean": np.mean(tt) / 60 if tt else float("nan"),
            "train_min_sd": np.std(tt) / 60 if tt else float("nan"),
            "train_mem_MB": np.nanmax([l["peak_train_mem_MB"] for l in logs]) if logs else float("nan"),
            "epochs_mean": np.mean([l["epochs_run"] for l in logs]) if logs else float("nan")}


C = cost_profile(); VALUES["cost"] = C
write_tex("tab_cost", "\n".join([
    f"Parameters (M) & {C['params_M']:.2f} & {5 * C['params_M']:.2f} \\\\",
    f"MACs (G) / FLOPs (G) & {C['GMACs']:.2f} / {C['GFLOPs']:.2f} & {5 * C['GMACs']:.2f} / {5 * C['GFLOPs']:.2f} \\\\",
    f"Weights, FP32 (MB) & {C['size_MB']:.1f} & {5 * C['size_MB']:.1f} \\\\",
    f"Inference latency, batch 1 (ms/image) & {C['lat_ms_bs1']:.1f} & {5 * C['lat_ms_bs1']:.1f} \\\\",
    f"Throughput, batch 32 (images/s) & {C['thr_bs32']:.0f} & {C['thr_bs32'] / 5:.0f} \\\\",
    f"Peak inference memory, batch 32 (MB) & {C['mem_bs32_MB']:.0f} & {C['mem_bs32_MB']:.0f} \\\\",
    f"Training time (min) & {C['train_min_mean']:.1f} $\\pm$ {C['train_min_sd']:.1f} & {5 * C['train_min_mean']:.1f} \\\\",
    f"Peak training memory, batch {CFG['BS']} (MB) & {C['train_mem_MB']:.0f} & {C['train_mem_MB']:.0f} \\\\",
    f"\\multicolumn{{3}}{{l}}{{\\footnotesize Hardware: {C['gpu']}; ensemble members run sequentially.}} \\\\"]))


# %% [15] Figures: confusion matrices, reliability diagrams, t-SNE, learning curves
def plot_cm(cm, classes, path, title):
    cm = np.asarray(cm); nm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.6, 3.2)); ax.imshow(nm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{nm[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if nm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, fontsize=7)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=8)
    fig.tight_layout(); fig.savefig(path); fig.savefig(path.replace(".pdf", ".png"), dpi=200); plt.close(fig)


def plot_rel(curves, path, title):
    fig, ax = plt.subplots(figsize=(4.2, 4.2)); ax.plot([0, 1], [0, 1], "--", c="grey", lw=1, label="Perfect calibration")
    edges = np.linspace(0, 1, 16)
    for lab, (y, p) in curves.items():
        conf, acc = p.max(1), p.argmax(1) == y; xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (conf > lo) & (conf <= hi)
            if s.sum() >= 3:
                xs.append(conf[s].mean()); ys.append(acc[s].mean())
        ax.plot(xs, ys, marker="o", ms=3, lw=1.3, label=f"{lab} (ECE={ece_score(y, p):.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=9); ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(path); fig.savefig(path.replace(".pdf", ".png"), dpi=200); plt.close(fig)


FIG = f"{OUT}/figures"
for exp, es, name, title in [("lbc_in", "lbc_test", "lbc_in_test", "LBC $\\rightarrow$ LBC test"),
                             ("lbc_in", "mal_tt", "lbc_in_target_test", "LBC $\\rightarrow$ Malhari 75%"),
                             ("lbc_in_da", "mal_tt", "lbc_in_da_target_test", "LBC + CORAL/MK-MMD $\\rightarrow$ Malhari"),
                             ("lbc_in_fs", "mal_tt", "lbc_in_fs_target_test", "Few-shot $\\rightarrow$ Malhari"),
                             ("merged_in", "merged_test", "merged_in_test", "Multi-domain $\\rightarrow$ merged test")]:
    R = get(exp, es)
    if not R:
        continue
    r = R[0]
    plot_cm(r[MAIN]["cm"], BETHESDA, f"{FIG}/{name}_cm.pdf", title)
    plot_rel({"Single model (fold 0)": (r["y"], softmax(load_seed(exp, seeds_of(exp)[0])[0][f"{es}__logits"], 1)),
              "Uniform average": (r["y"], r["probs"]["uniform_prob"]), "Entropy-weighted": (r["y"], r["probs"]["entropy_weight"]),
              "Entropy-weighted + TS": (r["y"], r["probs"][MAIN])}, f"{FIG}/{name}_reliability.pdf", title)
if CAS:
    plot_cm(CAS["cascade"]["cm"], BETHESDA, f"{FIG}/cascade_lbc_cm.pdf", "Two-stage cascade (LBC test)")


def tsne_figure():
    a, b = load_seed("lbc_in", 0), load_seed("lbc_in_da", 0)
    if a is None or b is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, R, title in [(axes[0], a, "Baseline (no adaptation)"), (axes[1], b, "CORAL + MK-MMD")]:
        X = np.concatenate([R[0]["lbc_test__feats"], R[0]["mal_tt__feats"]])
        dom = np.array(["LBC"] * len(R[0]["lbc_test__feats"]) + ["Malhari"] * len(R[0]["mal_tt__feats"]))
        yy = np.concatenate([R[0]["lbc_test__y"], R[0]["mal_tt__y"]])
        E = TSNE(n_components=2, perplexity=min(30, len(X) // 4), random_state=0, init="pca").fit_transform(X)
        for d, mk in [("LBC", "o"), ("Malhari", "^")]:
            for c in range(4):
                s = (dom == d) & (yy == c)
                ax.scatter(E[s, 0], E[s, 1], s=9, marker=mk, alpha=0.7, label=f"{d}-{BETHESDA[c]}")
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    axes[1].legend(fontsize=6, ncol=2, loc="best")
    fig.tight_layout(); fig.savefig(f"{FIG}/tsne_da.pdf"); fig.savefig(f"{FIG}/tsne_da.png", dpi=200); plt.close(fig)


tsne_figure()


def learning_curves():
    logs = sorted(glob.glob(f"{OUT}/runs/lbc_in/s0/f*/log.json"))
    if not logs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for f in logs:
        h = json.load(open(f))["history"]; k = os.path.basename(os.path.dirname(f))
        ax[0].plot([e["epoch"] for e in h], [e["ce"] for e in h], label=k)
        ax[1].plot([e["epoch"] for e in h], [e["val_f1"] for e in h], label=k)
    ax[0].set_title("Training loss (focal)"); ax[1].set_title("Validation macro-F1 (early-stopping fold)")
    for a in ax:
        a.set_xlabel("Epoch"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/learning_curves.pdf"); fig.savefig(f"{FIG}/learning_curves.png", dpi=200); plt.close(fig)


learning_curves()


# %% [16] Numbers for the manuscript placeholders + bundle
def tidy(o):
    if isinstance(o, dict):
        return {str(k): tidy(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [tidy(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


VALUES["status"] = json.load(open(f"{OUT}/status.json"))
VALUES["seeds_finished"] = {e: seeds_of(e) for e in ["lbc_in", "lbc_sip", "lbc_in_da", "lbc_sip_da", "lbc_in_fs", "lbc_sip_fs",
                                                      "merged_in", "merged_sip", "cascade_binary", "cascade_severity",
                                                      "base_resnet50", "base_densenet121", "base_effb3_nocbam"]}
json.dump(tidy(VALUES), open(f"{OUT}/results/values_for_manuscript.json", "w"), indent=1)


def fmt(k, pct=True):
    v = VALUES.get(k)
    return "not run yet" if v is None else (f"{v:.2f}%" if pct else f"{v:.4f}")


lines = ["VALUES TO PASTE INTO THE ORANGE [placeholders] OF THE REVISED MANUSCRIPT", "=" * 72,
         f"Status: all tasks finished = {VALUES['status']['all_tasks_done']}",
         f"Seeds finished per experiment: {VALUES['seeds_finished']}", "",
         f"Abstract/Sec 4.1  LBC test accuracy, ImageNet init : {fmt('acc_lbc_in_lbc_test_None')}",
         f"Abstract/Sec 4.1  LBC test accuracy, SIPaKMeD init : {fmt('acc_lbc_sip_lbc_test_None')}",
         f"Sec 4.1  LBC test macro-F1 ImageNet / SIPaKMeD      : {fmt('f1_lbc_in_lbc_test_None', False)} / {fmt('f1_lbc_sip_lbc_test_None', False)}",
         f"Abstract  Malhari 75% accuracy (LBC only, ImageNet): {fmt('acc_lbc_in_mal_tt_None')}",
         f"Sec 4.1  Malhari macro-AUC source-only (ImageNet)   : {fmt('auc_lbc_in_mal_tt_None', False)}",
         f"Sec 4.1  Few-shot macro-F1 base -> fine-tuned       : {fmt('f1_lbc_in_mal_tt_None', False)} -> {fmt('f1_lbc_in_fs_mal_tt_None', False)}",
         f"Abstract  Multi-domain LBC / Malhari test accuracy  : {fmt('acc_merged_in_merged_test_LBC')} / {fmt('acc_merged_in_merged_test_Malhari')}",
         f"Sec 4.1  Multi-domain merged test accuracy          : {fmt('acc_merged_in_merged_test_None')}",
         f"Sec 3.3  Number of labelled/unlabelled adapt images : {VALUES['n_Malhari_adapt']}",
         f"Sec 3.3  Duplicate audit                            : {VALUES['audit']}",
         f"Limitations  SCC images in LBC test set             : {VALUES['n_SCC_LBC_test']}",
         f"Sec 4.1  Malhari hold-out size (multi-domain)       : {VALUES['n_Malhari_merged_test']}", ""]
for k, v in VALUES.items():
    if k.startswith("cmp::"):
        lines.append(f"{k[5:]}: dAcc={100 * v['accuracy_diff']:+.2f} pp (t p={v['accuracy_t_p']:.3f}), dF1={v['f1_diff']:+.4f} "
                     f"(t p={v['f1_t_p']:.3f}, Wilcoxon p={v['f1_wilcoxon_p']:.3f}), dAUC={v['auc_diff']:+.4f}, "
                     f"McNemar Holm max p={v['mcnemar_holm_max']:.3f}, seed-0 bootstrap 95% CI of dF1={tuple(round(x, 4) for x in v['f1_diff_seed0_boot95'])}")
for k in ["tau_sweep_lbc_in_lbc_test", "errAUROC_lbc_in_lbc_test", "errAUROC_lbc_in_mal_tt", "error_all_vs_keep90_lbc_in_lbc_test",
          "error_all_vs_keep90_lbc_in_mal_tt", "ece_uniform_vs_entropyTS_lbc_in_lbc_test", "ece_uniform_vs_entropyTS_lbc_in_mal_tt",
          "cascade", "sipakmed_test", "cost"]:
    if k in VALUES:
        lines.append(f"{k}: {tidy(VALUES[k])}")
open(f"{OUT}/results/VALUES_FOR_MANUSCRIPT.txt", "w").write("\n".join(lines))
print("\n".join(lines))

with zipfile.ZipFile(f"{OUT}/manuscript_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for sub in ["generated", "figures", "results", "audit"]:
        for f in glob.glob(f"{OUT}/{sub}/*"):
            z.write(f, f"{sub}/{os.path.basename(f)}")
    for f in glob.glob(f"{OUT}/splits/*.csv"):
        z.write(f, f"splits/{os.path.basename(f)}")
log(f"bundle written: {OUT}/manuscript_bundle.zip")
if not VALUES["status"]["all_tasks_done"]:
    print("\nNOT FINISHED: add this notebook's output as an input and run again to continue.")
