"""Code-cell export from 05_ResNet50_V2_BreastCrop_512_v5.ipynb.

The executed notebook is the experiment record/source of truth.
This export is provided for code review and searchability.
"""

# %% [notebook cell 2]
from google.colab import drive
drive.mount("/content/drive")

# %% [notebook cell 3]
import os, json, random, platform, shutil
from datetime import datetime

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode          # FIX 2: bilinear augmentation
from PIL import Image, ImageEnhance
import matplotlib, matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy import ndimage
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             roc_auc_score, roc_curve)

PROJECT_DIR = "/content/drive/MyDrive/BreastCancer_dissertation"
DATA_DIR    = os.path.join(PROJECT_DIR, "Data", "CBIS-DDSM")
JPEG_DIR    = os.path.join(DATA_DIR, "jpeg")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "Results", "resnet50_v2_breastcrop_512")
MODEL_DIR   = os.path.join(PROJECT_DIR, "Models")
for d in (OUTPUT_DIR, MODEL_DIR): os.makedirs(d, exist_ok=True)

# FIX 6: local disk cache (Drive I/O + Otsu every epoch would starve the GPU)
# FIX (v4): cache key includes preprocessing settings, so changing CROP_PAD_FRAC or IMG_SIZE
# cannot silently reuse stale images preprocessed with the old settings.
PREPROCESS_VERSION  = "breastcrop_v4"
FORCE_REBUILD_CACHE = False        # set True if you change the preprocessing code itself

TRAIN_CSV, TEST_CSV = "mass_case_description_train_set.csv", "mass_case_description_test_set.csv"

SEED, IMG_SIZE, VAL_FRAC = 42, 512, 0.15
BATCH_SIZE, NUM_WORKERS = 8, 2
HEAD_EPOCHS, FINETUNE_EPOCHS, EARLY_STOPPING_PATIENCE = 4, 15, 5
HEAD_LR, FINETUNE_LR, WEIGHT_DECAY = 1e-3, 1e-5, 1e-4

MASK_BINARY_MIN, MASK_SIZE_TOL, ROI_CHECK_N = 0.60, 0.03, 250
BOOTSTRAP_N = 200          # ⚠️ SET TO 1000 FOR FINAL DISSERTATION NUMBERS (200 is for iteration only)
CROP_PAD_FRAC = 0.02       # raise if the lesion-retention check (§5) reports clipping

# versioned cache path (depends on the settings above)
CACHE_512_DIR = f"/content/cbis_cache_{PREPROCESS_VERSION}_{IMG_SIZE}_pad{int(CROP_PAD_FRAC*1000):03d}"
if FORCE_REBUILD_CACHE and os.path.exists(CACHE_512_DIR):
    shutil.rmtree(CACHE_512_DIR)
os.makedirs(CACHE_512_DIR, exist_ok=True)
print("cache dir:", CACHE_512_DIR)

# V1 artefacts — needed for the PAIRED comparison (FIX 4)
V1_MODEL_PATH   = os.path.join(MODEL_DIR, "resnet50_mass_baseline_patient_split.pth")
V1_METRICS_CSVS = [os.path.join(PROJECT_DIR,"Results","resnet50_mass_baseline_metrics.csv"),
                   os.path.join(PROJECT_DIR,"Results","resnet50_v1_metrics.csv")]

MODEL_PATH     = os.path.join(MODEL_DIR, "resnet50_v2_breastcrop_512_best.pth")
CONFIG_PATH    = os.path.join(OUTPUT_DIR, "config.json")
HISTORY_PATH   = os.path.join(OUTPUT_DIR, "training_history.csv")
SPLIT_PATH     = os.path.join(OUTPUT_DIR, "split_assignments.csv")
THRESHOLD_PATH = os.path.join(OUTPUT_DIR, "selected_threshold.json")
TEST_PRED_PATH = os.path.join(OUTPUT_DIR, "test_predictions.csv")
METRICS_PATH   = os.path.join(OUTPUT_DIR, "test_metrics.json")
COMPARISON_PATH= os.path.join(OUTPUT_DIR, "v1_v2_comparison.csv")
COVERAGE_PATH  = os.path.join(OUTPUT_DIR, "roi_mask_coverage_audit.csv")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = bool(torch.cuda.is_available())
print("Device:", device, "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("PyTorch", torch.__version__)

# %% [notebook cell 5]
def uid_from(p): return str(p).split("/")[-2]

def jpegs_from_metadata_path(mp):
    folder = os.path.join(JPEG_DIR, uid_from(mp))
    if not os.path.isdir(folder): return []
    return [os.path.join(folder,f) for f in sorted(os.listdir(folder))
            if f.lower().endswith((".jpg",".jpeg",".png"))]

def first_jpeg(mp):
    fs = jpegs_from_metadata_path(mp); return fs[0] if fs else None

def join_unique(vals):
    out=[]
    for v in vals:
        if isinstance(v,str) and v not in out: out.append(v)
    return "||".join(out)

ALLOWED_PATHOLOGY = {"BENIGN","BENIGN_WITHOUT_CALLBACK","MALIGNANT"}   # FIX: explicit mapping

def build_image_level_df(case_csv, split_name):
    raw = pd.read_csv(os.path.join(DATA_DIR, case_csv))
    observed = set(raw["pathology"].dropna().astype(str).str.upper())
    assert observed.issubset(ALLOWED_PATHOLOGY), f"Unexpected pathology values: {observed - ALLOWED_PATHOLOGY}"
    raw["label"] = (raw["pathology"].astype(str).str.upper()=="MALIGNANT").astype(int)
    raw["full_jpeg"] = raw["image file path"].apply(first_jpeg)
    missing, before = int(raw["full_jpeg"].isna().sum()), len(raw)

    df = (raw.dropna(subset=["full_jpeg"])
             .groupby(["patient_id","left or right breast","image view"], as_index=False)
             .agg(full_jpeg=("full_jpeg","first"), label=("label","max"),
                  pathology=("pathology", lambda x:"; ".join(sorted(set(map(str,x))))),
                  assessment=("assessment", lambda x:"; ".join(sorted(set(map(str,x))))),
                  breast_density=("breast_density","first"),
                  mass_shape=("mass shape", lambda x:"; ".join(sorted(set(map(str,x))))),
                  mass_margins=("mass margins", lambda x:"; ".join(sorted(set(map(str,x))))),
                  roi_mask_paths=("ROI mask file path", join_unique)))
    df["split_source"]=split_name
    df["sample_id"] = (df["patient_id"].astype(str)+"|"+df["left or right breast"].astype(str)
                       +"|"+df["image view"].astype(str))
    assert df["sample_id"].is_unique
    rep = {"split":split_name,"metadata_rows":before,"image_rows":len(df),
           "patients":int(df.patient_id.nunique()),"missing_files":missing,
           "malignant":int(df.label.sum()),"malignant_rate":float(df.label.mean())}
    return df.reset_index(drop=True), rep

img_df_train, rtr = build_image_level_df(TRAIN_CSV,"development")
img_df_test,  rte = build_image_level_df(TEST_CSV,"locked_test")
display(pd.DataFrame([rtr,rte]).round(4))
assert set(img_df_train.patient_id).isdisjoint(set(img_df_test.patient_id))

# %% [notebook cell 7]
sgkf = StratifiedGroupKFold(n_splits=max(2,round(1/VAL_FRAC)), shuffle=True, random_state=SEED)
tr_i, va_i = next(sgkf.split(img_df_train, img_df_train.label.to_numpy(), img_df_train.patient_id.to_numpy()))
train_df = img_df_train.iloc[tr_i].reset_index(drop=True); train_df["split"]="train"
val_df   = img_df_train.iloc[va_i].reset_index(drop=True); val_df["split"]="validation"
test_df  = img_df_test.copy();                             test_df["split"]="locked_test"
assert set(train_df.patient_id).isdisjoint(set(val_df.patient_id))
assert set(train_df.patient_id).isdisjoint(set(test_df.patient_id))
assert set(val_df.patient_id).isdisjoint(set(test_df.patient_id))
split_df = pd.concat([train_df,val_df,test_df], ignore_index=True); split_df.to_csv(SPLIT_PATH, index=False)
display(split_df.groupby("split").agg(images=("sample_id","count"), patients=("patient_id","nunique"),
        malignant=("label","sum"), malignant_rate=("label","mean")).reset_index().round(4))

# %% [notebook cell 9]
def otsu_threshold(g):
    h = np.bincount(g.ravel(), minlength=256).astype(np.float64); p = h/max(g.size,1)
    om = np.cumsum(p); mu = np.cumsum(p*np.arange(256)); mt = mu[-1]
    s = (mt*om-mu)**2/np.maximum(om*(1-om),1e-12); s[(om<=0)|(om>=1)]=0
    return int(np.argmax(s))

def breast_bbox(g, pad_frac=None):
    pad_frac = CROP_PAD_FRAC if pad_frac is None else pad_frac
    h,w = g.shape
    m = g > max(otsu_threshold(g),5)
    m = ndimage.binary_opening(m, structure=np.ones((3,3)))
    m = ndimage.binary_closing(m, structure=np.ones((7,7)))
    lab,n = ndimage.label(m)
    if n==0: return (0,0,w,h)
    c = np.bincount(lab.ravel()); c[0]=0
    ys,xs = np.where(lab==c.argmax())
    if len(xs)==0: return (0,0,w,h)
    pad = int(round(pad_frac*max(h,w)))
    return (max(int(xs.min())-pad,0), max(int(ys.min())-pad,0),
            min(int(xs.max())+1+pad,w), min(int(ys.max())+1+pad,h))

def _cpr(arr, bbox, size, resample):
    l,t,r,b = bbox
    crop = Image.fromarray(arr).crop((l,t,r,b)); cw,ch = crop.size; side = max(cw,ch)
    cv = Image.new("L",(side,side),0); cv.paste(crop,((side-cw)//2,(side-ch)//2))
    return np.array(cv.resize((size,size), resample))

def crop_pad_resize_image(g,bbox,size=IMG_SIZE): return _cpr(g,bbox,size,Image.BILINEAR).astype(np.uint8)
def crop_pad_resize_mask(m,bbox,size=IMG_SIZE):  return (_cpr(m,bbox,size,Image.NEAREST)>127).astype(np.uint8)
def load_gray(p): return np.array(Image.open(p).convert("L"), dtype=np.uint8)

def preprocess_mammogram(path):
    o = load_gray(path); bb = breast_bbox(o)
    return o, crop_pad_resize_image(o,bb), bb

# LEAKAGE FIX: inspect DEVELOPMENT images only. Never look at the locked test set before
# training — tuning CROP_PAD_FRAC on what you saw there would be test-set tuning.
ex = train_df.sample(min(5, len(train_df)), random_state=SEED).reset_index(drop=True)
fig,ax = plt.subplots(len(ex),3,figsize=(10,3.2*len(ex)))
for r,row in ex.iterrows():
    o,p,bb = preprocess_mammogram(row.full_jpeg); l,t,rr,b = bb
    ax[r,0].imshow(o,cmap="gray"); ax[r,0].set_title("original")
    ax[r,1].imshow(o[t:b,l:rr],cmap="gray"); ax[r,1].set_title("breast crop")
    ax[r,2].imshow(p,cmap="gray"); ax[r,2].set_title("padded 512")
    for a in ax[r]: a.axis("off")
plt.tight_layout(); plt.show()

# %% [notebook cell 11]
def full_size_binary_mask(row, full_shape, tol=MASK_SIZE_TOL, binary_min=MASK_BINARY_MIN):
    """Binary lesion mask registered to the FULL mammogram.
    Rejects crop-level masks and grayscale lesion crops.
    BUGFIX: masks within tolerance but not pixel-identical are resized before union
    (np.maximum on mismatched shapes would raise). Also handles inverted (white-background) masks."""
    fh, fw = full_shape
    union = np.zeros((fh, fw), dtype=np.uint8); found = False
    for mp in str(row.get("roi_mask_paths", "")).split("||"):
        for f in jpegs_from_metadata_path(mp):
            a = load_gray(f); h, w = a.shape

            # reject crop-level images (cannot be registered to the full mammogram)
            if abs(h-fh)/max(fh,1) > tol or abs(w-fw)/max(fw,1) > tol:
                continue
            # reject grayscale lesion crops (not a binary mask)
            if float(np.mean((a < 20) | (a > 235))) < binary_min:
                continue

            # BUGFIX: align small dimension differences before combining
            if (h, w) != (fh, fw):
                a = np.array(Image.fromarray(a).resize((fw, fh), Image.NEAREST))

            m = a > 127
            # some masks are stored as black lesion on white background -> invert
            if m.mean() > 0.5:
                m = ~m
            m = m.astype(np.uint8)
            if m.sum() == 0:
                continue

            union = np.maximum(union, m); found = True
    return union if (found and union.sum() > 0) else None

# LEAKAGE FIX: audit on TRAINING data only. CROP_PAD_FRAC / MASK_BINARY_MIN may be adjusted
# from what this shows, so it must not see the locked test set.
audit = train_df.sample(min(ROI_CHECK_N, len(train_df)), random_state=SEED)

# coverage sweep
rows=[]
sweep = audit.sample(min(120,len(audit)), random_state=SEED)
for bmin in [0.50,0.60,0.70,0.80,0.90,0.95]:
    ok = sum(full_size_binary_mask(r, load_gray(r.full_jpeg).shape, binary_min=bmin) is not None
             for _,r in sweep.iterrows())
    rows.append({"binary_min":bmin,"coverage":ok/len(sweep)})
    print(f"binary_min={bmin:.2f} -> coverage {ok/len(sweep):.1%}")
pd.DataFrame(rows).to_csv(COVERAGE_PATH, index=False)

# lesion-retention check
ret=[]; n_full=0
for _,row in tqdm(audit.iterrows(), total=len(audit), desc="lesion retention"):
    o,_,bb = preprocess_mammogram(row.full_jpeg)
    m = full_size_binary_mask(row, o.shape)
    if m is None: continue
    n_full += 1; l,t,r,b = bb
    if m.sum(): ret.append(m[t:b,l:r].sum()/m.sum())

print(f"\nFull-image masks in audit: {n_full}/{len(audit)}")
if ret:
    ret=np.array(ret)
    print(f"lesion pixels retained -> mean {ret.mean():.4f} | min {ret.min():.4f} | fully kept {(ret>=0.999).mean():.1%}")
    if ret.min() < 0.999: print("[!] crop is clipping lesion tissue -> raise CROP_PAD_FRAC and re-run")
else:
    print("[!] No full-image masks. CBIS-DDSM likely stores CROP-level masks for these rows.")
    print("    -> Phase 5 attention supervision needs a different registration source. Do NOT compute Dice/IoU/attention loss yet.")

# %% [notebook cell 13]
ex2=[]
for _,row in audit.iterrows():
    o,p,bb = preprocess_mammogram(row.full_jpeg)
    m = full_size_binary_mask(row,o.shape)
    if m is None: continue
    ex2.append((row,p,crop_pad_resize_mask(m*255,bb)))
    if len(ex2)>=6: break

if ex2:
    fig,ax = plt.subplots(len(ex2),2,figsize=(8,3.2*len(ex2)))
    if len(ex2)==1: ax=np.array([ax])
    for r,(row,p,m) in enumerate(ex2):
        ax[r,0].imshow(p,cmap="gray"); ax[r,0].set_title("processed 512"); ax[r,0].axis("off")
        ax[r,1].imshow(p,cmap="gray"); ax[r,1].contour(m,levels=[.5],colors="red",linewidths=1.3)
        ax[r,1].set_title("malignant" if row.label else "benign"); ax[r,1].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"roi_alignment.png"),dpi=200,bbox_inches="tight"); plt.show()
    print("GATE: contours must sit ON the lesion. If not, do not build the attention loss on these masks.")
else:
    print("No alignable masks to visualise — see verdict above.")

# %% [notebook cell 15]
def cache_path(sid): return os.path.join(CACHE_512_DIR, sid.replace("|","_")+".png")

def build_cache(df, name):
    todo = [(r.sample_id, r.full_jpeg) for r in df.itertuples() if not os.path.exists(cache_path(r.sample_id))]
    for sid, path in tqdm(todo, desc=f"caching {name}"):
        _,proc,_ = preprocess_mammogram(path)
        Image.fromarray(proc).save(cache_path(sid))
for d,n in [(train_df,"train"),(val_df,"val"),(test_df,"test")]: build_cache(d,n)
print("cached 512 images:", len(os.listdir(CACHE_512_DIR)))

# %% [notebook cell 17]
def augment(img):
    if random.random()<0.5: img = TF.hflip(img)
    if random.random()<0.8:
        dx = int(0.03*img.size[0]); dy = int(0.03*img.size[1])
        img = TF.affine(img, angle=random.uniform(-8,8),
                        translate=(random.randint(-dx,dx), random.randint(-dy,dy)),
                        scale=1.0, shear=[0.,0.],
                        interpolation=InterpolationMode.BILINEAR, fill=0)
    if random.random()<0.5: img = ImageEnhance.Brightness(img).enhance(random.uniform(0.90,1.10))
    if random.random()<0.5: img = ImageEnhance.Contrast(img).enhance(random.uniform(0.90,1.15))
    return img

IMAGENET_MEAN = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

class MammogramDataset(Dataset):
    def __init__(self, df, train=False): self.df=df.reset_index(drop=True); self.train=train
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        row = self.df.iloc[i]
        img = Image.open(cache_path(row.sample_id)).convert("RGB")   # from local cache
        if self.train: img = augment(img)
        t = torch.from_numpy(np.array(img,dtype=np.float32)/255.).permute(2,0,1)
        t = (t-IMAGENET_MEAN)/IMAGENET_STD
        return {"image":t.float(), "label":torch.tensor(float(row.label),dtype=torch.float32),
                "patient_id":str(row.patient_id), "sample_id":str(row.sample_id), "path":str(row.full_jpeg)}

def seed_worker(worker_id):
    # derive from torch's per-worker seed: reproducible, but augmentation randomness advances properly
    ws = torch.initial_seed() % (2**32)
    np.random.seed(ws); random.seed(ws)
g = torch.Generator(); g.manual_seed(SEED)
train_loader = DataLoader(MammogramDataset(train_df,True), batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, worker_init_fn=seed_worker, generator=g)
val_loader   = DataLoader(MammogramDataset(val_df),  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader  = DataLoader(MammogramDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
print(f"batches: train {len(train_loader)} | val {len(val_loader)} | test {len(test_loader)}")

# %% [notebook cell 19]
def build_model():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features,1)
    return m

def freeze_batchnorm_stats(m):
    """Batch size 8 -> noisy BN stats corrupt pretrained features.
    Freeze running stats AND the affine (scale/bias) params: with ~1.2k images this is the safe choice."""
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()
            for prm in mod.parameters(): prm.requires_grad = False

def freeze_backbone(m):
    for n,p in m.named_parameters(): p.requires_grad = n.startswith("fc")
def unfreeze_layer4(m):
    for n,p in m.named_parameters(): p.requires_grad = n.startswith("layer4") or n.startswith("fc")

pos = float((train_df.label==1).sum()); neg = float((train_df.label==0).sum())
pos_weight = torch.tensor([neg/max(pos,1.)], device=device)
model = build_model().to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
print("pos_weight:", round(pos_weight.item(),4))

# %% [notebook cell 21]
def to_device(b): return b["image"].to(device,non_blocking=True), b["label"].to(device,non_blocking=True).view(-1,1)

def predict_loader(m, loader):
    m.eval(); recs=[]
    with torch.no_grad():
        for b in loader:
            x,y = to_device(b); p = torch.sigmoid(m(x)).cpu().numpy().ravel()
            for sid,pid,pa,yy,pp in zip(b["sample_id"],b["patient_id"],b["path"],y.cpu().numpy().ravel(),p):
                recs.append({"sample_id":sid,"patient_id":pid,"path":pa,"y_true":int(yy),"p_malignant":float(pp)})
    return pd.DataFrame(recs)

def eval_loss(m, loader):
    m.eval(); tot=0.; n=0
    with torch.no_grad():
        for b in loader:
            x,y = to_device(b); tot += criterion(m(x),y).item()*x.size(0); n += x.size(0)
    return tot/max(n,1)

def metrics_from(y,p,th=0.5):
    y=np.asarray(y).astype(int); p=np.asarray(p,dtype=float); pr=(p>=th).astype(int)
    tn,fp,fn,tp = confusion_matrix(y,pr,labels=[0,1]).ravel()
    two = len(np.unique(y))==2
    return {"roc_auc":float(roc_auc_score(y,p)) if two else np.nan,
            "average_precision":float(average_precision_score(y,p)) if two else np.nan,
            "accuracy":float(accuracy_score(y,pr)),"balanced_accuracy":float(balanced_accuracy_score(y,pr)),
            "sensitivity":float(tp/(tp+fn)) if (tp+fn) else np.nan,
            "specificity":float(tn/(tn+fp)) if (tn+fp) else np.nan,
            "precision":float(precision_score(y,pr,zero_division=0)),
            "f1":float(f1_score(y,pr,zero_division=0)),
            "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"threshold":float(th)}

def train_epoch(m, loader, opt, scaler):
    m.train(); freeze_batchnorm_stats(m)                      # FIX 1
    tot=0.; n=0; amp = device.type=="cuda"
    for b in tqdm(loader, leave=False):
        x,y = to_device(b); opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            loss = criterion(m(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot += loss.item()*x.size(0); n += x.size(0)
    return tot/max(n,1)

def run_stage(name, m, epochs, opt, sched=None, start=0):
    hist=[]; best=-np.inf; best_state=None; patience=EARLY_STOPPING_PATIENCE
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))
    for e in range(1, epochs+1):
        tl = train_epoch(m, train_loader, opt, scaler)
        vl = eval_loss(m, val_loader)
        vp = predict_loader(m, val_loader); vm = metrics_from(vp.y_true, vp.p_malignant)
        if sched is not None: sched.step(vm["roc_auc"])
        hist.append({"stage":name,"epoch":start+e,"train_loss":tl,"val_loss":vl,
                     "val_roc_auc":vm["roc_auc"],"val_accuracy":vm["accuracy"],"lr":opt.param_groups[0]["lr"]})
        print(f"{name} ep{start+e:02d} | train {tl:.4f} | val {vl:.4f} | val AUC {vm['roc_auc']:.4f}")
        if vm["roc_auc"] > best:
            best=vm["roc_auc"]; best_state={k:v.detach().cpu() for k,v in m.state_dict().items()}
            patience=EARLY_STOPPING_PATIENCE
        else:
            patience -= 1
            if patience<=0: print(f"early stop ({name})"); break
    if best_state is not None: m.load_state_dict(best_state)
    return hist, best

# %% [notebook cell 23]
history=[]; best_auc=-np.inf; best_state=None
freeze_backbone(model)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
h,b = run_stage("head", model, HEAD_EPOCHS, opt); history+=h
if b>best_auc: best_auc,best_state = b,{k:v.detach().cpu() for k,v in model.state_dict().items()}

unfreeze_layer4(model)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
h,b = run_stage("layer4_finetune", model, FINETUNE_EPOCHS, opt, sched, start=len(history)); history+=h
if b>best_auc: best_auc,best_state = b,{k:v.detach().cpu() for k,v in model.state_dict().items()}

model.load_state_dict(best_state); torch.save(best_state, MODEL_PATH)
hist_df = pd.DataFrame(history); hist_df.to_csv(HISTORY_PATH, index=False)
print("Best val AUC:", round(best_auc,4))
display(hist_df.round(4))

fig,ax = plt.subplots(1,2,figsize=(11,4))
ax[0].plot(hist_df.epoch, hist_df.train_loss, label="train"); ax[0].plot(hist_df.epoch, hist_df.val_loss, label="val")
ax[0].set_title("loss (overfitting check)"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(hist_df.epoch, hist_df.val_roc_auc); ax[1].set_title("validation AUC"); ax[1].set_xlabel("epoch")
plt.tight_layout(); plt.show()

# %% [notebook cell 25]
vp = predict_loader(model, val_loader)
rows=[{"threshold":th, **{k:metrics_from(vp.y_true,vp.p_malignant,th)[k] for k in ["balanced_accuracy","sensitivity","specificity"]}}
      for th in np.linspace(0.01,0.99,99)]
th_df = pd.DataFrame(rows); th_df["youden_j"] = th_df.sensitivity + th_df.specificity - 1
selected_threshold = float(th_df.sort_values(["youden_j","balanced_accuracy"],ascending=False).iloc[0].threshold)
json.dump({"method":"validation_youden_j","threshold":selected_threshold}, open(THRESHOLD_PATH,"w"), indent=2)
print("Selected threshold (VALIDATION only):", round(selected_threshold,3))

# %% [notebook cell 27]
def bootstrap_ci_patient_level(pred_df, th, n_boot=BOOTSTRAP_N, seed=SEED):
    """Resample PATIENTS, not images: a patient's mammograms are not independent."""
    rng = np.random.default_rng(seed)
    by_patient = {pid: grp for pid, grp in pred_df.groupby("patient_id")}
    pids = np.array(list(by_patient))
    recs=[]
    for _ in range(n_boot):
        draw = rng.choice(pids, size=len(pids), replace=True)
        bs = pd.concat([by_patient[p] for p in draw], ignore_index=True)
        if bs.y_true.nunique() < 2: continue
        recs.append(metrics_from(bs.y_true, bs.p_malignant, th))
    bt = pd.DataFrame(recs)
    return {k:[float(bt[k].quantile(.025)), float(bt[k].quantile(.975))]
            for k in ["roc_auc","sensitivity","specificity","balanced_accuracy","f1"]}

model.load_state_dict(torch.load(MODEL_PATH, map_location=device)); model.eval()
test_pred = predict_loader(model, test_loader); test_pred.to_csv(TEST_PRED_PATH, index=False)

m05  = metrics_from(test_pred.y_true, test_pred.p_malignant, 0.5)
msel = metrics_from(test_pred.y_true, test_pred.p_malignant, selected_threshold)
ci   = bootstrap_ci_patient_level(test_pred, selected_threshold)
json.dump({"threshold_0_5":m05,"validation_selected":msel,"ci95_patient_level":ci},
          open(METRICS_PATH,"w"), indent=2)
display(pd.DataFrame([{"setting":"threshold_0.5",**m05},
                      {"setting":"validation_selected",**msel}]).round(4))
print("\n95% patient-level CIs @ selected threshold:")
for k,v in ci.items(): print(f"  {k:20s} [{v[0]:.3f}, {v[1]:.3f}]")

# %% [notebook cell 28]
y=test_pred.y_true.to_numpy(); p=test_pred.p_malignant.to_numpy()
fpr,tpr,_ = roc_curve(y,p); pr,rc,_ = precision_recall_curve(y,p)
cm = confusion_matrix(y,(p>=selected_threshold).astype(int),labels=[0,1])
fig,ax = plt.subplots(1,3,figsize=(15,4.2))
ax[0].plot(fpr,tpr,label=f"AUC={msel['roc_auc']:.3f}"); ax[0].plot([0,1],[0,1],"--",c="gray")
ax[0].set_title("ROC"); ax[0].legend(); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
ax[1].plot(rc,pr); ax[1].set_title("Precision-Recall"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
ax[2].imshow(cm,cmap="Blues"); ax[2].set_title("Confusion matrix")
ax[2].set_xticks([0,1]); ax[2].set_xticklabels(["not malig","malig"])
ax[2].set_yticks([0,1]); ax[2].set_yticklabels(["not malig","malig"])
for i in range(2):
    for j in range(2): ax[2].text(j,i,cm[i,j],ha="center",color="white" if cm[i,j]>cm.max()/2 else "black")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"test_curves.png"),dpi=200,bbox_inches="tight"); plt.show()

# %% [notebook cell 30]
class V1Dataset(Dataset):
    """V1 preprocessing: raw mammogram squashed to 224, no breast crop."""
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        row = self.df.iloc[i]
        img = Image.open(row.full_jpeg).convert("L").resize((224,224))
        t = torch.from_numpy(np.stack([np.array(img,dtype=np.float32)/255.]*3,0))
        return {"image":((t-IMAGENET_MEAN)/IMAGENET_STD).float(),
                "label":torch.tensor(float(row.label),dtype=torch.float32),
                "patient_id":str(row.patient_id), "sample_id":str(row.sample_id), "path":str(row.full_jpeg)}

v1_pred = None
if os.path.exists(V1_MODEL_PATH):
    v1 = models.resnet50(weights=None)
    sd = torch.load(V1_MODEL_PATH, map_location=device)
    out_dim = sd["fc.weight"].shape[0] if "fc.weight" in sd else 2
    v1.fc = nn.Linear(v1.fc.in_features, out_dim)
    v1.load_state_dict(sd); v1 = v1.to(device).eval()

    loader = DataLoader(V1Dataset(test_df), batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
    recs=[]
    with torch.no_grad():
        for b in loader:
            x,_ = to_device(b); logits = v1(x)
            p = (torch.softmax(logits,1)[:,1] if out_dim==2 else torch.sigmoid(logits[:,0])).cpu().numpy()
            for sid,pid,yy,pp in zip(b["sample_id"],b["patient_id"],b["label"].numpy().ravel(),p):
                recs.append({"sample_id":sid,"patient_id":pid,"y_true":int(yy),"p_malignant":float(pp)})
    v1_pred = pd.DataFrame(recs)
    print("Scored V1 on the same test images:", len(v1_pred))
else:
    print("[!] V1 checkpoint not found at", V1_MODEL_PATH)
    print("    -> cannot run the paired test. Report V1 vs V2 descriptively only.")

# %% [notebook cell 31]
rows=[]
if v1_pred is not None:
    v1_m05 = metrics_from(v1_pred.y_true, v1_pred.p_malignant, 0.5)
    rows.append({"model":"ResNet-50 V1","image_size":"224","preprocessing":"direct resize",
                 "threshold":"0.5", **{k:v1_m05[k] for k in ["roc_auc","accuracy","balanced_accuracy","sensitivity","specificity","f1"]}})
rows.append({"model":"ResNet-50 V2","image_size":"512","preprocessing":"breast crop + pad","threshold":"0.5",
             **{k:m05[k] for k in ["roc_auc","accuracy","balanced_accuracy","sensitivity","specificity","f1"]}})
rows.append({"model":"ResNet-50 V2","image_size":"512","preprocessing":"breast crop + pad","threshold":"validation Youden J",
             **{k:msel[k] for k in ["roc_auc","accuracy","balanced_accuracy","sensitivity","specificity","f1"]}})
comp = pd.DataFrame(rows); comp.to_csv(COMPARISON_PATH, index=False); display(comp.round(4))

# ---- paired patient-level bootstrap of the AUC DIFFERENCE ----
if v1_pred is not None:
    merged = (v1_pred[["sample_id","patient_id","y_true","p_malignant"]]
              .rename(columns={"p_malignant":"p_v1"})
              .merge(test_pred[["sample_id","p_malignant"]].rename(columns={"p_malignant":"p_v2"}), on="sample_id"))
    assert len(merged)==len(test_pred), "V1/V2 predictions do not cover the same samples"
    rng = np.random.default_rng(SEED)
    by_p = {pid:grp for pid,grp in merged.groupby("patient_id")}; pids = np.array(list(by_p))
    diffs=[]
    for _ in range(max(BOOTSTRAP_N, 1000)):
        draw = rng.choice(pids, size=len(pids), replace=True)
        bs = pd.concat([by_p[p] for p in draw], ignore_index=True)
        if bs.y_true.nunique()<2: continue
        diffs.append(roc_auc_score(bs.y_true,bs.p_v2) - roc_auc_score(bs.y_true,bs.p_v1))
    diffs=np.array(diffs)
    obs = roc_auc_score(merged.y_true,merged.p_v2) - roc_auc_score(merged.y_true,merged.p_v1)
    lo,hi = np.quantile(diffs,[.025,.975])
    print(f"\nPAIRED AUC difference (V2 - V1): {obs:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    if lo > 0:   print("CI excludes 0 -> V2's improvement is unlikely to be test-set noise.")
    elif hi < 0: print("CI excludes 0 but is NEGATIVE -> V2 is significantly WORSE. Investigate crops/training.")
    else:        print("CI includes 0 -> the difference is within noise. Report descriptively; do not claim improvement.")

# %% [notebook cell 33]
tw = test_df.merge(test_pred[["sample_id","p_malignant"]], on="sample_id", how="left")
tw["y_true"]=tw.label.astype(int); tw["y_pred"]=(tw.p_malignant>=selected_threshold).astype(int)
tw["outcome"]=np.select([(tw.y_true==1)&(tw.y_pred==1),(tw.y_true==0)&(tw.y_pred==0),
                         (tw.y_true==0)&(tw.y_pred==1),(tw.y_true==1)&(tw.y_pred==0)],
                        ["true_positive","true_negative","false_positive","false_negative"])
display(tw.outcome.value_counts())

cases=[]
for oc in ["true_positive","true_negative","false_positive","false_negative"]:
    sub = tw[tw.outcome==oc].copy()
    if len(sub)==0: continue
    sub["conf"] = (sub.p_malignant - selected_threshold).abs()      # FIX 5
    cases += sub.sort_values("conf",ascending=False).head(2).index.tolist()

if not cases:
    print("No eligible TP/TN/FP/FN cases found.")
else:
    fig,ax = plt.subplots(len(cases),2,figsize=(8,3.3*len(cases)))
    if len(cases)==1: ax=np.array([ax])
    for r,idx in enumerate(cases):
        row = tw.loc[idx]; o,p,bb = preprocess_mammogram(row.full_jpeg)
        m = full_size_binary_mask(row,o.shape)
        ax[r,0].imshow(p,cmap="gray"); ax[r,0].set_title("processed"); ax[r,0].axis("off")
        ax[r,1].imshow(p,cmap="gray")
        if m is not None: ax[r,1].contour(crop_pad_resize_mask(m*255,bb),levels=[.5],colors="red",linewidths=1.2)
        ax[r,1].set_title("ROI (if aligned)"); ax[r,1].axis("off")
        ax[r,0].text(0,-0.12, f"{row.outcome} | true={row.y_true} pred={row.y_pred} p={row.p_malignant:.3f}\n"
                     f"density={row.get('breast_density','NA')} | shape={row.get('mass_shape','NA')} | margin={row.get('mass_margins','NA')}",
                     transform=ax[r,0].transAxes, fontsize=9, va="top")
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"error_analysis.png"),dpi=200,bbox_inches="tight"); plt.show()

# %% [notebook cell 36]
def cam_input(proc):
    a = np.dstack([proc]*3).astype(np.float32)/255.
    t = torch.from_numpy(a).permute(2,0,1)
    return ((t-IMAGENET_MEAN)/IMAGENET_STD).unsqueeze(0).to(device)

def compute_gradcam(proc, target_class=1):
    """target_class=1 -> malignant evidence; 0 -> benign evidence (negated logit).
    Explaining the PREDICTED class means TN/FN cases are explained properly too."""
    model.eval(); acts=[]; grads=[]
    layer = model.layer4[-1]
    h1 = layer.register_forward_hook(lambda m,i,o: acts.append(o))
    h2 = layer.register_full_backward_hook(lambda m,gi,go: grads.append(go[0]))
    model.zero_grad(set_to_none=True)
    logit = model(cam_input(proc))[:,0]
    (logit if target_class == 1 else -logit).sum().backward()
    w = grads[0].mean(dim=(2,3),keepdim=True)
    cam = F.relu((w*acts[0]).sum(1,keepdim=True))
    cam = F.interpolate(cam,size=(IMG_SIZE,IMG_SIZE),mode="bilinear",align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy(); cam -= cam.min()
    if cam.max()>0: cam /= cam.max()
    h1.remove(); h2.remove(); return cam

def overlay(g,cam,alpha=.45):
    gg = g.astype(float)/255.; rgb = np.dstack([gg,gg,gg])
    return np.clip((1-alpha)*rgb + alpha*matplotlib.colormaps["jet"](cam)[:,:,:3],0,1)

gc = cases[:8]
if gc:
    fig,ax = plt.subplots(len(gc),3,figsize=(12,3.5*len(gc)))
    if len(gc)==1: ax=np.array([ax])
    for r,idx in enumerate(gc):
        row = tw.loc[idx]; o,p,bb = preprocess_mammogram(row.full_jpeg)
        cam = compute_gradcam(p, target_class=int(row.y_pred))   # explain the PREDICTED class
        m = full_size_binary_mask(row,o.shape)
        ax[r,0].imshow(p,cmap="gray"); ax[r,0].set_title("processed"); ax[r,0].axis("off")
        ax[r,1].imshow(overlay(p,cam)); ax[r,1].set_title(f"Grad-CAM ({'malignant' if row.y_pred==1 else 'benign'} evidence)"); ax[r,1].axis("off")
        ax[r,2].imshow(p,cmap="gray")
        if m is not None: ax[r,2].contour(crop_pad_resize_mask(m*255,bb),levels=[.5],colors="red",linewidths=1.2)
        ax[r,2].set_title("ROI (if aligned)"); ax[r,2].axis("off")
        ax[r,0].text(0,-0.12,f"{row.outcome} | p={row.p_malignant:.3f}",transform=ax[r,0].transAxes,fontsize=9,va="top")
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"gradcam.png"),dpi=200,bbox_inches="tight"); plt.show()

# %% [notebook cell 38]
json.dump({"created_at":datetime.now().isoformat(),"seed":SEED,"image_size":IMG_SIZE,
           "batch_size":BATCH_SIZE,"head_epochs":HEAD_EPOCHS,"finetune_epochs":FINETUNE_EPOCHS,
           "head_lr":HEAD_LR,"finetune_lr":FINETUNE_LR,"weight_decay":WEIGHT_DECAY,
           "crop_pad_frac":CROP_PAD_FRAC,"mask_binary_min":MASK_BINARY_MIN,
           "bootstrap_n":BOOTSTRAP_N,"selected_threshold":selected_threshold,
           "frozen_batchnorm":True,"device":str(device)}, open(CONFIG_PATH,"w"), indent=2)
print("artefacts ->", OUTPUT_DIR)

