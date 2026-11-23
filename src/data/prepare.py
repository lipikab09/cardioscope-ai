from pathlib import Path
import argparse,json
import numpy as np, wfdb
from scipy.signal import butter,sosfiltfilt
from .labels import map_symbol,CLASS_TO_IDX
from .splits import TRAIN,VAL,DS2,EXCLUDED,validate_splits

def prepare_record(root,rec,fs=360,low=.5,high=40,pre=.25,post=.45):
    sig,fields=wfdb.rdsamp(str(Path(root)/rec)); ann=wfdb.rdann(str(Path(root)/rec),"atr")
    names=[s.upper().replace(" ","") for s in fields["sig_name"]]
    lead=next((i for i,n in enumerate(names) if "MLII" in n),next((i for i,n in enumerate(names) if "II" in n),0))
    x=sig[:,lead].astype(np.float32)
    x=sosfiltfilt(butter(4,[low,high],btype="bandpass",fs=fs,output="sos"),x).astype(np.float32)
    left,right=int(pre*fs),int(post*fs); X=[]; y=[]
    for pos,sym in zip(ann.sample,ann.symbol):
        label=map_symbol(sym)
        if label is None: continue
        a,b=int(pos)-left,int(pos)+right
        if a<0 or b>len(x): continue
        beat=x[a:b]; med=np.median(beat); scale=np.percentile(np.abs(beat-med),95)+1e-6
        X.append(np.clip((beat-med)/scale,-5,5)); y.append(CLASS_TO_IDX[label])
    return np.asarray(X,dtype=np.float32),np.asarray(y,dtype=np.int64),fields["sig_name"][lead]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",default="data/mitdb"); ap.add_argument("--output",default="data/processed"); a=ap.parse_args()
    validate_splits(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    manifest={"train":TRAIN,"validation":VAL,"test":DS2,"excluded":EXCLUDED}
    for split,records in [("train",TRAIN),("val",VAL),("test",DS2)]:
        xs=[]; ys=[]; rr=[]
        for rec in records:
            x,y,lead=prepare_record(a.data_dir,rec); xs.append(x); ys.append(y); rr.extend([rec]*len(y))
            print(split,rec,len(y),"beats",lead)
        np.savez_compressed(out/f"{split}.npz",X=np.concatenate(xs),y=np.concatenate(ys),record=np.asarray(rr,dtype="U3"))
    (out/"split_manifest.json").write_text(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
