import argparse,json,random
from pathlib import Path
import yaml,numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from .models import CardioScopeNet
from .losses import FocalLoss
from .metrics import classification_metrics,patient_bootstrap_macro_f1
from .data.labels import CLASSES
class Beats(Dataset):
    def __init__(self,path):
        d=np.load(path); self.x=torch.tensor(d["X"][:,None,:],dtype=torch.float32); self.y=torch.tensor(d["y"],dtype=torch.long); self.records=d["record"]
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.x[i],self.y[i]
def predict(model,loader,device):
    model.eval(); yy=[]; pp=[]
    with torch.no_grad():
        for x,y in loader: pp.extend(torch.softmax(model(x.to(device)),1).cpu().numpy()); yy.extend(y.numpy())
    return np.asarray(yy),np.asarray(pp)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml"); a=p.parse_args()
    cfg=yaml.safe_load(open(a.config)); seed=cfg["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device="cuda" if cfg["training"]["device"]=="auto" and torch.cuda.is_available() else "cpu"
    tr=Beats(Path(cfg["data"]["processed"])/"train.npz"); va=Beats(Path(cfg["data"]["processed"])/"val.npz"); te=Beats(Path(cfg["data"]["processed"])/"test.npz")
    train=DataLoader(tr,batch_size=cfg["training"]["batch_size"],shuffle=True); val=DataLoader(va,batch_size=512); test=DataLoader(te,batch_size=512)
    weights=compute_class_weight("balanced",classes=np.arange(5),y=tr.y.numpy())
    model=CardioScopeNet(dropout=cfg["model"]["dropout"]).to(device); lossfn=FocalLoss(torch.tensor(weights,dtype=torch.float32,device=device),cfg["training"]["focal_gamma"])
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",patience=3,factor=.5)
    best=-1; stale=0; out=Path("artifacts"); out.mkdir(exist_ok=True)
    for ep in range(cfg["training"]["epochs"]):
        model.train(); total=0
        for x,y in train:
            x=x.to(device); y=y.to(device); opt.zero_grad(); loss=lossfn(model(x),y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2); opt.step(); total+=loss.item()
        vy,vp=predict(model,val,device); score=f1_score(vy,vp.argmax(1),average="macro",zero_division=0); sched.step(score)
        print(f"epoch {ep+1} loss={total/len(train):.4f} validation_macro_f1={score:.4f}")
        if score>best:
            best=score; stale=0; torch.save({"model":model.state_dict(),"config":cfg},out/"cardioscope_best.pt")
        else:
            stale+=1
            if stale>=cfg["training"]["patience"]: break
    ck=torch.load(out/"cardioscope_best.pt",map_location=device,weights_only=False); model.load_state_dict(ck["model"])
    y,probs=predict(model,test,device); m=classification_metrics(y,probs,CLASSES); m["patient_bootstrap_95ci_macro_f1"]=patient_bootstrap_macro_f1(y,probs.argmax(1),te.records)
    m["test_records"]=sorted(set(te.records.tolist())); (out/"test_metrics.json").write_text(json.dumps(m,indent=2))
    np.savez_compressed(out/"test_predictions.npz",y=y,probs=probs,records=te.records)
    print("Held-out metrics:",{k:v for k,v in m.items() if k not in ("report","confusion_matrix")})
if __name__=="__main__": main()
