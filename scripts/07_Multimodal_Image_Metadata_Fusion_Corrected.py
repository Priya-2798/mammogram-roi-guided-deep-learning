"""Code-cell export from 07_Multimodal_Image_Metadata_Fusion_Corrected.ipynb.

The executed notebook is the experiment record/source of truth.
This export is provided for code review and searchability.
"""

# %% [notebook cell 2]
from google.colab import drive
drive.mount("/content/drive")

# %% [notebook cell 3]
import os, json, random, shutil
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy import ndimage
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, balanced_accuracy_score, confusion_matrix)

PROJECT_DIR="/content/drive/MyDrive/BreastCancer_dissertation"
DATA_DIR=os.path.join(PROJECT_DIR,"Data","CBIS-DDSM"); JPEG_DIR=os.path.join(DATA_DIR,"jpeg")
OUTPUT_DIR=os.path.join(PROJECT_DIR,"Results","multimodal_fusion"); MODEL_DIR=os.path.join(PROJECT_DIR,"Models")
for d in (OUTPUT_DIR,MODEL_DIR): os.makedirs(d,exist_ok=True)
def find_csv(n):
    for c in (os.path.join(DATA_DIR,n),os.path.join(DATA_DIR,"csv",n)):
        if os.path.exists(c): return c
    raise FileNotFoundError(n)
V2_SPLIT_CSV=os.path.join(PROJECT_DIR,"Results","resnet50_v2_breastcrop_512","split_assignments.csv")
ROI_CKPT=os.path.join(MODEL_DIR,"resnet50_cam2_lambda0.5.pth")   # M4 image-branch init

TRAIN_CSV,TEST_CSV="mass_case_description_train_set.csv","mass_case_description_test_set.csv"
META_COLS=["breast_density","mass_shape","mass_margins"]         # BI-RADS 'assessment' EXCLUDED on purpose
SEED,IMG_SIZE,BATCH_SIZE,NUM_WORKERS=42,512,8,0
HEAD_EPOCHS,FINETUNE_EPOCHS,META_EPOCHS,PATIENCE=4,15,40,6
HEAD_LR,FINETUNE_LR,META_LR,WEIGHT_DECAY=1e-3,1e-5,1e-3,1e-4
CROP_PAD_FRAC=0.02; BOOTSTRAP_N=1000
CACHE_DIR=f"/content/cbis_mm_512_pad{int(CROP_PAD_FRAC*1000):03d}"; os.makedirs(os.path.join(CACHE_DIR,"img"),exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

# %% [notebook cell 5]
def uid_from(p): return str(p).split("/")[-2]
def jpegs_from(mp):
    f=os.path.join(JPEG_DIR,uid_from(mp))
    return [os.path.join(f,x) for x in sorted(os.listdir(f))] if os.path.isdir(f) else []
def first_jpeg(mp):
    fs=[x for x in jpegs_from(mp) if x.lower().endswith((".jpg",".jpeg",".png"))]; return fs[0] if fs else None

def clean_meta_value(v):
    if pd.isna(v): return "UNKNOWN"
    s=str(v).strip()
    return s if s and s.lower() not in {"nan","none"} else "UNKNOWN"

def first_non_missing(vals):
    for v in vals:
        s=clean_meta_value(v)
        if s != "UNKNOWN": return s
    return "UNKNOWN"

def combine_unique(vals):
    # Label-independent lesion descriptor aggregation. Compound CBIS-DDSM strings are preserved;
    # only multiple abnormality rows are joined with '|'.
    out=[]
    for v in vals:
        s=clean_meta_value(v)
        if s != "UNKNOWN" and s not in out: out.append(s)
    return "|".join(sorted(out)) if out else "UNKNOWN"

def build_df(case_csv):
    raw=pd.read_csv(find_csv(case_csv))
    raw["label"]=(raw["pathology"].astype(str).str.upper()=="MALIGNANT").astype(int)
    raw["full_jpeg"]=raw["image file path"].apply(first_jpeg)

    df=(raw.dropna(subset=["full_jpeg"])
          .groupby(["patient_id","left or right breast","image view"],as_index=False,sort=False)
          .agg(full_jpeg=("full_jpeg","first"), label=("label","max"),
               breast_density=("breast_density",first_non_missing),
               mass_shape=("mass shape",combine_unique), mass_margins=("mass margins",combine_unique)))
    df["sample_id"]=(df.patient_id.astype(str)+"|"+df["left or right breast"].astype(str)+"|"+df["image view"].astype(str))
    assert df["sample_id"].is_unique, "Image-level sample IDs must be unique."
    return df.reset_index(drop=True)

train_all=build_df(TRAIN_CSV); test_df=build_df(TEST_CSV)

# reuse EXACT V2 split (strict)
if not os.path.exists(V2_SPLIT_CSV): raise FileNotFoundError("Run V2 first — need its split_assignments.csv")
v2=pd.read_csv(V2_SPLIT_CSV); v2["split"]=v2.split.astype(str).str.lower()
tr_ids=set(v2[v2.split=="train"].sample_id); va_ids=set(v2[v2.split=="validation"].sample_id)
test_ids=set(v2[v2["split"].isin(["locked_test","test"])]["sample_id"])
train_df=train_all[train_all.sample_id.isin(tr_ids)].reset_index(drop=True)
val_df  =train_all[train_all.sample_id.isin(va_ids)].reset_index(drop=True)

assert set(train_df["sample_id"]) == tr_ids, "Train split sample mismatch with saved V2 split."
assert set(val_df["sample_id"]) == va_ids, "Validation split sample mismatch with saved V2 split."
if test_ids:
    assert set(test_df["sample_id"]) == test_ids, (
        f"Test split mismatch: current={len(test_df)}, saved V2={len(test_ids)}"
    )

for a,b in [(train_df,val_df),(train_df,test_df),(val_df,test_df)]:
    assert set(a.patient_id).isdisjoint(set(b.patient_id))
print("Exact V2 train, validation and test samples confirmed.")
print(f"train {len(train_df)} | val {len(val_df)} | test {len(test_df)}  (V2 split reused)")

# %% [notebook cell 7]
metadata_columns=["breast_density","mass_shape","mass_margins"]
for col in metadata_columns:
    print(f"\n{col}"); print(train_df[col].value_counts(dropna=False))
print("\n--- missing values per split ---")
for split_name,df in {"train":train_df,"validation":val_df,"test":test_df}.items():
    print(split_name); print(df[metadata_columns].isna().sum())

# %% [notebook cell 9]
def norm_value(x):
    return clean_meta_value(x)

def tokens_from_value(x):
    s=norm_value(x)
    if s == "UNKNOWN": return ["UNKNOWN"]
    return [t for t in str(s).split("|") if t]

def fit_encoder(df, cols):
    cats={}
    for c in cols:
        vals=set()
        for x in df[c]: vals.update(tokens_from_value(x))
        vals.add("UNKNOWN")
        cats[c]=sorted(vals)
    return cats

def encode(df, cats, cols):
    blocks=[]
    for c in cols:
        idx={x:i for i,x in enumerate(cats[c])}
        mh=np.zeros((len(df),len(cats[c])),np.float32)
        unknown_i=idx["UNKNOWN"]
        for j,x in enumerate(df[c]):
            toks=[t if t in idx else "UNKNOWN" for t in tokens_from_value(x)]
            if not toks: toks=["UNKNOWN"]
            for tok in toks: mh[j,idx.get(tok,unknown_i)]=1.0
        blocks.append(mh)
    return np.concatenate(blocks,1)

CATS=fit_encoder(train_df, META_COLS)             # FIT ON TRAIN ONLY
META_DIM=sum(len(CATS[c]) for c in META_COLS)
meta_np={"train":encode(train_df,CATS,META_COLS),"val":encode(val_df,CATS,META_COLS),"test":encode(test_df,CATS,META_COLS)}
json.dump({c:CATS[c] for c in META_COLS}, open(os.path.join(OUTPUT_DIR,"metadata_categories.json"),"w"), indent=2)
print("metadata multi-hot dim:",META_DIM,"| categories/col:",{c:len(CATS[c]) for c in META_COLS})
# map sample_id -> meta vector
meta_lookup={}
for split,df in [("train",train_df),("val",val_df),("test",test_df)]:
    for i,sid in enumerate(df.sample_id): meta_lookup[sid]=meta_np[split][i]

# %% [notebook cell 11]
def otsu(g):
    h=np.bincount(g.ravel(),minlength=256).astype(float); p=h/max(g.size,1); om=np.cumsum(p); mu=np.cumsum(p*np.arange(256)); mt=mu[-1]
    s=(mt*om-mu)**2/np.maximum(om*(1-om),1e-12); s[(om<=0)|(om>=1)]=0; return int(np.argmax(s))
def bbox(g,pad=CROP_PAD_FRAC):
    h,w=g.shape; m=g>max(otsu(g),5); m=ndimage.binary_opening(m,structure=np.ones((3,3))); m=ndimage.binary_closing(m,structure=np.ones((7,7)))
    lab,n=ndimage.label(m)
    if n==0: return (0,0,w,h)
    c=np.bincount(lab.ravel()); c[0]=0; ys,xs=np.where(lab==c.argmax())
    if len(xs)==0: return (0,0,w,h)
    q=int(round(pad*max(h,w)))
    return (max(int(xs.min())-q,0),max(int(ys.min())-q,0),min(int(xs.max())+1+q,w),min(int(ys.max())+1+q,h))
def crop_img(path):
    g=np.array(Image.open(path).convert("L"),np.uint8); l,t,r,b=bbox(g)
    cr=Image.fromarray(g).crop((l,t,r,b)); cw,ch=cr.size; sd=max(cw,ch); cv=Image.new("L",(sd,sd),0); cv.paste(cr,((sd-cw)//2,(sd-ch)//2))
    return np.array(cv.resize((IMG_SIZE,IMG_SIZE),Image.BILINEAR),np.uint8)
def ip(sid): return os.path.join(CACHE_DIR,"img",sid.replace("|","_")+".png")
for df in (train_df,val_df,test_df):
    for r in tqdm(df.itertuples(),total=len(df),desc="cache"):
        if not os.path.exists(ip(r.sample_id)): Image.fromarray(crop_img(r.full_jpeg)).save(ip(r.sample_id))
print("cached:",len(os.listdir(os.path.join(CACHE_DIR,"img"))))

# %% [notebook cell 13]
MEAN=torch.tensor([0.485,0.456,0.406]).view(3,1,1); STD=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
def aug(img):
    if random.random()<0.5: img=TF.hflip(img)
    if random.random()<0.8:
        dx=int(.03*img.size[0]); dy=int(.03*img.size[1])
        img=TF.affine(img,angle=random.uniform(-8,8),translate=(random.randint(-dx,dx),random.randint(-dy,dy)),
                      scale=1.,shear=[0.,0.],interpolation=InterpolationMode.BILINEAR,fill=0)
    if random.random()<0.5: img=ImageEnhance.Brightness(img).enhance(random.uniform(.9,1.1))
    if random.random()<0.5: img=ImageEnhance.Contrast(img).enhance(random.uniform(.9,1.15))
    return img
class MMDS(Dataset):
    def __init__(self,df,train=False): self.df=df.reset_index(drop=True); self.train=train
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; img=Image.open(ip(r.sample_id)).convert("L")
        if self.train: img=aug(img)
        t=torch.from_numpy(np.stack([np.array(img,np.float32)/255.]*3,0)); t=(t-MEAN)/STD
        return {"image":t.float(),"meta":torch.tensor(meta_lookup[r.sample_id],dtype=torch.float32),
                "label":torch.tensor(float(r.label)),"sample_id":r.sample_id,"patient_id":str(r.patient_id)}
def sw(w): s=torch.initial_seed()%(2**32); np.random.seed(s); random.seed(s)
def loader(df,train,shuf,seed=SEED):
    g=torch.Generator(); g.manual_seed(seed)
    return DataLoader(MMDS(df,train),batch_size=BATCH_SIZE,shuffle=shuf,num_workers=NUM_WORKERS,
                      pin_memory=(device.type=="cuda"),worker_init_fn=sw,generator=g)
def make_train_loader(seed=SEED):
    # Fresh loader per model: same shuffle order for M1-M4 comparisons.
    return loader(train_df,True,True,seed=seed)
val_loader=loader(val_df,False,False); test_loader=loader(test_df,False,False)
print(f"batches: train {len(make_train_loader())} | val {len(val_loader)} | test {len(test_loader)}")

# %% [notebook cell 15]
class ImageBranch(nn.Module):
    def __init__(self, roi_ckpt=None):
        super().__init__()
        bb=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features=nn.Sequential(bb.conv1,bb.bn1,bb.relu,bb.maxpool,bb.layer1,bb.layer2,bb.layer3,bb.layer4)
        self.out_dim=2048
        if roi_ckpt:
            sd=torch.load(roi_ckpt,map_location="cpu")
            fsd={k[len("features."):]:v for k,v in sd.items() if k.startswith("features.")}
            self.features.load_state_dict(fsd); print("M4: loaded ROI-supervised backbone from",os.path.basename(roi_ckpt))
    def forward(self,x): return F.adaptive_avg_pool2d(self.features(x),1).flatten(1)
class MetaMLP(nn.Module):
    def __init__(self,in_dim,out_dim=64):
        super().__init__(); self.out_dim=out_dim
        self.net=nn.Sequential(nn.Linear(in_dim,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,out_dim),nn.ReLU())
    def forward(self,x): return self.net(x)
class MMModel(nn.Module):
    def __init__(self,use_image,use_meta,meta_in,roi_ckpt=None):
        super().__init__(); self.use_image=use_image; self.use_meta=use_meta; d=0
        if use_image: self.image=ImageBranch(roi_ckpt); d+=self.image.out_dim
        if use_meta:  self.meta=MetaMLP(meta_in);       d+=self.meta.out_dim
        self.head=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,1))
    def forward(self,image,meta):
        f=[]
        if self.use_image: f.append(self.image(image))
        if self.use_meta:  f.append(self.meta(meta))
        return self.head(torch.cat(f,1))

def freeze_bn(m):
    for mod in m.modules():
        if isinstance(mod,nn.BatchNorm2d):
            mod.eval()
            for p in mod.parameters(): p.requires_grad=False
def set_trainable(m,stage):
    for n,p in m.named_parameters():
        if n.startswith("image.features.7"): p.requires_grad=(stage=="finetune")
        elif n.startswith("image.features"): p.requires_grad=False
        else: p.requires_grad=True

# %% [notebook cell 17]
pos=float((train_df.label==1).sum()); neg=float((train_df.label==0).sum())
POS_W=torch.tensor([neg/max(pos,1.)],device=device); criterion=nn.BCEWithLogitsLoss(pos_weight=POS_W)

@torch.no_grad()
def predict(m,loader):
    m.eval(); rows=[]
    for b in loader:
        logit=m(b["image"].to(device),b["meta"].to(device))
        p=torch.sigmoid(logit).cpu().numpy().ravel()
        for sid,pid,y,pp in zip(b["sample_id"],b["patient_id"],b["label"].numpy(),p):
            rows.append({"sample_id":sid,"patient_id":pid,"y_true":int(y),"p_malignant":float(pp)})
    return pd.DataFrame(rows)
def metrics_cls(y,p,th):
    y=np.asarray(y).astype(int); p=np.asarray(p,float); pr=(p>=th).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pr,labels=[0,1]).ravel()
    return {"roc_auc":roc_auc_score(y,p),"pr_auc":average_precision_score(y,p),
            "sensitivity":tp/(tp+fn) if (tp+fn) else np.nan,"specificity":tn/(tn+fp) if (tn+fp) else np.nan,
            "balanced_acc":balanced_accuracy_score(y,pr),"f1":f1_score(y,pr,zero_division=0)}
def pick_threshold(vp):
    best=(0.5,-1)
    for th in np.linspace(0.05,0.95,91):
        mm=metrics_cls(vp.y_true,vp.p_malignant,th); j=mm["sensitivity"]+mm["specificity"]-1
        if j>best[1]: best=(th,j)
    return best[0]

def clone_state(m): return {k:v.detach().cpu() for k,v in m.state_dict().items()}

def run_stage(m,epochs,opt,sched,uses_image,train_loader_local):
    best=-np.inf; best_state=None; patience=PATIENCE
    scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda"))
    for e in range(1,epochs+1):
        m.train()
        if uses_image: freeze_bn(m)
        tot=0.;n=0
        for b in tqdm(train_loader_local,leave=False):
            x=b["image"].to(device); mt=b["meta"].to(device); y=b["label"].view(-1,1).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=(device.type=="cuda")):
                loss=criterion(m(x,mt),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot+=loss.item()*x.size(0); n+=x.size(0)
        train_loss=tot/max(n,1)
        vp=predict(m,val_loader); auc=roc_auc_score(vp.y_true,vp.p_malignant)
        print(f"Epoch {e:02d}/{epochs} | train loss {train_loss:.4f} | val AUC {auc:.4f}")
        if sched: sched.step(auc)
        if auc>best:
            best=auc; best_state=clone_state(m); patience=PATIENCE
        else:
            patience-=1
            if patience<=0: break
    if best_state is not None: m.load_state_dict(best_state)
    return best,best_state

def freeze_image_branch(m):
    if hasattr(m,"image"):
        for p in m.image.parameters(): p.requires_grad=False

def train_model(name, use_image, use_meta, roi_ckpt=None, freeze_roi_backbone=False):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    train_loader_local=make_train_loader(SEED)
    m=MMModel(use_image,use_meta,META_DIM,roi_ckpt).to(device)
    best_auc=-np.inf; best_state=None
    if use_image and freeze_roi_backbone:
        # M4: ROI-supervised backbone initialisation + metadata fusion.
        # Freeze the ROI image branch so classification-only fine-tuning cannot erase ROI-guided features.
        freeze_image_branch(m)
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=META_LR,weight_decay=WEIGHT_DECAY)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=3)
        auc,state=run_stage(m,META_EPOCHS,opt,sched,True,train_loader_local)
        best_auc,best_state=auc,state
    elif use_image:
        set_trainable(m,"head")
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=HEAD_LR,weight_decay=WEIGHT_DECAY)
        auc,state=run_stage(m,HEAD_EPOCHS,opt,None,True,train_loader_local)
        if auc>best_auc: best_auc,best_state=auc,state
        set_trainable(m,"finetune")
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=FINETUNE_LR,weight_decay=WEIGHT_DECAY)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=2)
        auc,state=run_stage(m,FINETUNE_EPOCHS,opt,sched,True,train_loader_local)
        if auc>best_auc: best_auc,best_state=auc,state
    else:
        for p in m.parameters(): p.requires_grad=True
        opt=torch.optim.AdamW(m.parameters(),lr=META_LR,weight_decay=WEIGHT_DECAY)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=3)
        auc,state=run_stage(m,META_EPOCHS,opt,sched,False,train_loader_local)
        best_auc,best_state=auc,state
    assert best_state is not None, f"No validation checkpoint was produced for {name}"
    m.load_state_dict(best_state)
    path=os.path.join(MODEL_DIR,f"mm_{name}.pth"); torch.save(best_state,path)
    print(f"{name}: saved best validation AUC checkpoint ({best_auc:.4f})")
    return m,path

# %% [notebook cell 19]
import os
specs=[("M1_image_only",True,False,None,False),
       ("M2_metadata_only",False,True,None,False),
       ("M3_image_meta",True,True,None,False),
       ("M4_roi_meta",True,True,ROI_CKPT if os.path.exists(ROI_CKPT) else None,True)]
if not os.path.exists(ROI_CKPT): print("WARNING: ROI checkpoint not found -> M4 will use ImageNet init (run notebook 06 first for the real M4).")
print("M4 definition: ROI-supervised backbone initialisation plus metadata fusion; ROI image branch frozen for this first experiment.")
paths={}; val_preds={}
for name,ui,um,ck,freeze_roi in specs:
    print(f"\n===== {name} =====")
    m,path=train_model(name,ui,um,ck,freeze_roi_backbone=freeze_roi); paths[name]=path; val_preds[name]=predict(m,val_loader)
    val_preds[name].to_csv(os.path.join(OUTPUT_DIR,f"val_pred_{name}.csv"),index=False)
    del m; torch.cuda.empty_cache()
print("\ntrained:",list(paths))

# %% [notebook cell 21]
def load(name,ui,um,ck):
    m=MMModel(ui,um,META_DIM,ck if (ck and os.path.exists(ck)) else None).to(device)
    m.load_state_dict(torch.load(paths[name],map_location=device)); m.eval(); return m

test_preds={}; rows=[]
for name,ui,um,ck,freeze_roi in specs:
    m=load(name,ui,um,ck); tp=predict(m,test_loader); test_preds[name]=tp
    th=pick_threshold(val_preds[name]); cm=metrics_cls(tp.y_true,tp.p_malignant,th)
    tp.to_csv(os.path.join(OUTPUT_DIR,f"test_pred_{name}.csv"),index=False)
    rows.append({"model":name,"threshold":round(th,3),**cm}); del m; torch.cuda.empty_cache()
abl=pd.DataFrame(rows)[["model","threshold","roc_auc","pr_auc","sensitivity","specificity","balanced_acc","f1"]]
abl.to_csv(os.path.join(OUTPUT_DIR,"ablation_table.csv"),index=False)
print("=== ABLATION (locked test) ==="); display(abl.round(4))

# %% [notebook cell 23]
def boot_ci(pred,th,metric="roc_auc",n=BOOTSTRAP_N,seed=SEED):
    rng=np.random.default_rng(seed); by={k:g for k,g in pred.groupby("patient_id")}; pids=np.array(list(by)); vals=[]
    for _ in range(n):
        bs=pd.concat([by[p] for p in rng.choice(pids,len(pids),replace=True)],ignore_index=True)
        if bs.y_true.nunique()<2: continue
        vals.append(metrics_cls(bs.y_true,bs.p_malignant,th)[metric])
    return float(np.nanquantile(vals,.025)),float(np.nanquantile(vals,.975))
def paired_auc_delta(pa,pb,n=BOOTSTRAP_N,seed=SEED):
    mg=pa.rename(columns={"p_malignant":"pa"}).merge(pb[["sample_id","p_malignant"]].rename(columns={"p_malignant":"pb"}),on="sample_id")
    rng=np.random.default_rng(seed); by={k:g for k,g in mg.groupby("patient_id")}; pids=np.array(list(by)); ds=[]
    for _ in range(n):
        bs=pd.concat([by[p] for p in rng.choice(pids,len(pids),replace=True)],ignore_index=True)
        if bs.y_true.nunique()<2: continue
        ds.append(roc_auc_score(bs.y_true,bs.pa)-roc_auc_score(bs.y_true,bs.pb))
    return float(roc_auc_score(mg.y_true,mg.pa)-roc_auc_score(mg.y_true,mg.pb)),float(np.quantile(ds,.025)),float(np.quantile(ds,.975))

print("AUC with 95% patient-level CI:")
for r in rows:
    lo,hi=boot_ci(test_preds[r["model"]],r["threshold"]); print(f"  {r['model']:18s} AUC {r['roc_auc']:.4f}  CI [{lo:.3f},{hi:.3f}]")
print("\nKEY COMPARISONS (paired patient-level bootstrap of ΔAUC):")
for lab,a,b in [("metadata value: M3 - M1","M3_image_meta","M1_image_only"),
                ("ROI-feature value: M4 - M3","M4_roi_meta","M3_image_meta")]:
    o,l,h=paired_auc_delta(test_preds[a],test_preds[b])
    print(f"  {lab}: {o:+.4f}  95% CI [{l:+.4f},{h:+.4f}]  ({'excludes 0' if (l>0 or h<0) else 'includes 0'})")

