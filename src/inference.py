from pathlib import Path
import numpy as np,torch
from scipy.signal import butter,sosfiltfilt
from .models import CardioScopeNet
from .data.labels import CLASSES
def load_model(path="artifacts/cardioscope_best.pt"):
    device="cuda" if torch.cuda.is_available() else "cpu"; ck=torch.load(path,map_location=device,weights_only=False)
    model=CardioScopeNet(dropout=ck.get("config",{}).get("model",{}).get("dropout",.25)).to(device); model.load_state_dict(ck["model"]); model.eval()
    return model,device
def preprocess_signal(signal,fs=360):
    x=np.asarray(signal,dtype=np.float32).squeeze()
    if x.ndim!=1: raise ValueError("Provide one ECG lead as a 1-D array.")
    if fs<=80: raise ValueError("Sampling rate must exceed 80 Hz.")
    return sosfiltfilt(butter(4,[.5,40],btype="bandpass",fs=fs,output="sos"),x)
def predict_window(model,window,device):
    x=np.asarray(window,dtype=np.float32); med=np.median(x); x=np.clip((x-med)/(np.percentile(np.abs(x-med),95)+1e-6),-5,5)
    with torch.no_grad(): p=torch.softmax(model(torch.tensor(x[None,None],dtype=torch.float32,device=device)),1)[0].cpu().numpy()
    return {CLASSES[i]:float(p[i]) for i in range(5)}
