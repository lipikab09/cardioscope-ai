from pathlib import Path
import json
import numpy as np,pandas as pd,streamlit as st,plotly.graph_objects as go,plotly.express as px
from src.data.labels import CLASSES
from src.data.splits import TRAIN,VAL,DS2
from src.inference import load_model,preprocess_signal,predict_window
from src.explain import grad_cam_1d
import torch

st.set_page_config(page_title="CardioScope AI",page_icon="🫀",layout="wide")
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono&family=Manrope:wght@400;500;700;800&display=swap');
html,body,[class*="css"]{font-family:Manrope,sans-serif}
.stApp{background:radial-gradient(ellipse at 10% 0%,#153447 0%,#07131e 48%,#050b12 100%);color:#e9f6fa}
[data-testid="stSidebar"]{background:#091722;border-right:1px solid #244354}
.hero{padding:30px;border:1px solid #284b5b;border-radius:22px;background:linear-gradient(120deg,#12394a,#0b1b29);margin-bottom:18px}
.eyebrow{font:11px 'DM Mono';letter-spacing:2px;color:#63e6d1}.hero h1{font-size:42px;color:#f1fbff;letter-spacing:-2px}.hero p,.muted{color:#a7c0cc}
.panel{padding:18px;border:1px solid #254555;border-radius:16px;background:#0b1c29;margin:8px 0}
div[data-testid="stMetric"]{background:#0b1c29;border:1px solid #254555;padding:12px;border-radius:14px}
.stButton>button{background:#63e6d1;color:#07131e;border:0;border-radius:10px;font-weight:800}
</style>''',unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## 🫀 CardioScope AI")
    st.caption("SIGNAL INTELLIGENCE / RESEARCH LAB")
    page=st.radio("NAVIGATION",["Overview","Data protocol","Architecture","Evaluation","Signal explorer","Inference","Research notes"])
    st.markdown("---"); st.caption("PATIENT-DISJOINT • EXPLAINABLE AI")
def hero(k,t,d): st.markdown(f'<div class="hero"><div class="eyebrow">{k}</div><h1>{t}</h1><p>{d}</p></div>',unsafe_allow_html=True)
def panel(t,d): st.markdown(f'<div class="panel"><h3>{t}</h3><p class="muted">{d}</p></div>',unsafe_allow_html=True)
mp=Path("artifacts/cardioscope_best.pt"); metrics=Path("artifacts/test_metrics.json")
if page=="Overview":
    hero("ECG / DEEP LEARNING / TRUSTWORTHY EVALUATION","Beyond accuracy. Toward trustworthy ECG AI.","A patient-independent arrhythmia research pipeline with residual 1D CNNs, AAMI-style labels, explainability and robustness analysis.")
    c=st.columns(4)
    for col,(v,l) in zip(c,[("48","MIT-BIH records"),("360 Hz","sampling frequency"),("5 + 2","AAMI + binary classes"),("0","patient overlap across splits")]):
        col.metric(l,v)
    panel("The core question","Can a heartbeat classifier generalise to a person it has never seen? Random beat splits can leak patient-specific morphology across train and test. CardioScope keeps records together.")
    panel("Pipeline","WFDB → lead selection → 0.5–40 Hz filtering → annotation-centred beat windows → patient-disjoint split → residual CNN → evaluation + explanation.")
    if mp.exists(): st.success("Trained model checkpoint detected.")
    else: st.info("No trained checkpoint yet. Run the training commands in README; the dashboard will not fabricate scores.")
    st.error("Research prototype only. Not for diagnosis, treatment, triage or emergency decisions.")
elif page=="Data protocol":
    hero("01 / DATA GOVERNANCE","The split is part of the science.","Records 201 and 202 belong to the same subject; 202 is excluded to avoid cross-partition subject leakage.")
    cols=st.columns(3)
    for col,(name,rec) in zip(cols,[("TRAIN",TRAIN),("VALIDATION",VAL),("TEST",DS2)]):
        col.markdown(f"### {name} · {len(rec)} records"); col.write(", ".join(rec))
    st.markdown("### Preprocessing")
    st.code("WFDB signal + expert annotations\n→ MLII lead (fallback II)\n→ 4th-order zero-phase Butterworth 0.5–40 Hz\n→ 0.25 s before / 0.45 s after beat marker\n→ robust per-beat normalisation\n→ ignore non-beat annotations")
    st.dataframe(pd.DataFrame({"AAMI class":CLASSES,"Meaning":["Normal","Supraventricular ectopic","Ventricular ectopic","Fusion","Unknown / other"],"Symbols":["N,L,R,e,j","A,a,J,S","V,E","F","/,f,Q"]}),hide_index=True,use_container_width=True)
    st.caption("Binary view maps N to Normal and S/V/F/Q to Abnormal. This label collapse is not a clinical diagnosis.")
elif page=="Architecture":
    hero("02 / MODEL DESIGN","Residual morphology learning.","Temporal feature extraction with attention pooling.")
    for i,(a,b) in enumerate([("Input","1 × 252 samples"),("Stem","Conv1D 32 channels, 9-tap kernel"),("Residual block 1","48 channels, stride 2"),("Residual block 2","80 channels, stride 2"),("Residual block 3","128 channels, stride 2"),("Attention pooling","Learned temporal weighting + global average"),("Classifier head","256 → 96 → 5 logits")]):
        panel(f"{i+1:02d} / {a}",b)
    st.markdown("**Training:** AdamW • class-weighted focal loss • gradient clipping • ReduceLROnPlateau • early stopping on validation macro-F1.")
    st.warning("Beat windows use reference annotation locations. The pipeline does not yet evaluate an automated R-peak detector.")
elif page=="Evaluation":
    hero("03 / EVIDENCE","Measure what matters.","Inspect per-class performance rather than relying on accuracy alone.")
    if not metrics.exists(): st.warning("No metrics yet. Prepare the database and train the model first.")
    else:
        m=json.loads(metrics.read_text()); a,b,c,d=st.columns(4)
        a.metric("Accuracy",f"{m['accuracy']:.3f}"); b.metric("Balanced accuracy",f"{m['balanced_accuracy']:.3f}"); c.metric("Macro F1",f"{m['macro_f1']:.3f}"); d.metric("Weighted F1",f"{m['weighted_f1']:.3f}")
        fig=px.imshow(np.array(m["confusion_matrix"]),x=CLASSES,y=CLASSES,text_auto=True,color_continuous_scale="Teal",labels={"x":"Predicted","y":"Reference","color":"Beats"})
        fig.update_layout(template="plotly_dark",paper_bgcolor="#0b1c29",plot_bgcolor="#0b1c29")
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(pd.DataFrame(m["report"]).T.loc[CLASSES],use_container_width=True)
        ci=m.get("patient_bootstrap_95ci_macro_f1")
        if ci: st.info(f"Patient-cluster bootstrap 95% CI for macro-F1: {ci[0]:.3f}–{ci[1]:.3f}")
elif page=="Signal explorer":
    hero("04 / SIGNAL LAB","Explore the waveform.","Synthetic educational waveform with interactive baseline drift and noise controls.")
    n=252;t=np.arange(n)/360
    g=lambda mu,s,a:a*np.exp(-.5*((t-mu)/s)**2)
    clean=g(.18,.018,.12)-g(.29,.009,.15)+g(.32,.012,1)-g(.35,.014,.25)+g(.48,.045,.28)
    noise=st.slider("Noise amplitude",0.,.5,.05,.01); drift=st.slider("Baseline wander",0.,.5,.1,.01)
    sig=clean+noise*np.random.default_rng(4).normal(size=n)+drift*np.sin(2*np.pi*1.2*t)
    fig=go.Figure(); fig.add_scatter(x=t,y=clean,name="Clean synthetic beat"); fig.add_scatter(x=t,y=sig,name="Corrupted beat")
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0b1c29",plot_bgcolor="#0b1c29",xaxis_title="Time (s)",yaxis_title="Amplitude")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Synthetic illustration only — not patient data and not a model prediction.")
elif page=="Inference":
    hero("05 / INFERENCE WORKSPACE","Inspect one beat at a time.","Upload a CSV containing one ECG lead. Select a 252-sample window for an experimental model prediction.")
    if not mp.exists(): st.warning("Train the model before using inference.")
    else:
        up=st.file_uploader("Upload ECG CSV",type=["csv"])
        if up:
            df=pd.read_csv(up); col=st.selectbox("Lead column",df.columns.tolist()); fs=st.number_input("Sampling rate (Hz)",100,2000,360,10)
            sig=preprocess_signal(df[col].dropna().to_numpy(),int(fs))
            fig=go.Figure(go.Scatter(y=sig)); fig.update_layout(template="plotly_dark",height=260)
            st.plotly_chart(fig,use_container_width=True)
            if len(sig)>=252:
                start=st.slider("Window start",0,len(sig)-252,0)
                if st.button("Classify window"):
                    model,device=load_model(); probs=predict_window(model,sig[start:start+252],device)
                    predicted=max(probs,key=probs.get)
                    st.metric("Predicted AAMI-style class",predicted)
                    st.bar_chart(pd.Series(probs).sort_values(ascending=False))
                    x=torch.tensor(((sig[start:start+252]-np.median(sig[start:start+252]))/(np.percentile(np.abs(sig[start:start+252]-np.median(sig[start:start+252])),95)+1e-6))[None,None,:],dtype=torch.float32,device=device)
                    cam=grad_cam_1d(model,x)
                    explain=go.Figure()
                    explain.add_scatter(y=sig[start:start+252],name="ECG beat",line=dict(color="#63e6d1"))
                    explain.add_scatter(y=cam,name="Grad-CAM attribution",yaxis="y2",line=dict(color="#ffb86b"))
                    explain.update_layout(template="plotly_dark",paper_bgcolor="#0b1c29",plot_bgcolor="#0b1c29",title="Temporal attribution (Grad-CAM)",yaxis=dict(title="ECG amplitude"),yaxis2=dict(title="Attribution",overlaying="y",side="right",range=[0,1]))
                    st.plotly_chart(explain,use_container_width=True)
                    st.caption("Raw model scores are not calibrated clinical probabilities. Grad-CAM is a model-attribution aid, not a clinical explanation. The selected window may not be R-peak-centred.")
            else: st.error("Signal must contain at least 252 samples.")
elif page=="Research notes":
    hero("06 / TRANSPARENCY","Limitations are part of the result.","A credible ML project documents what it has not demonstrated.")
    panel("Current limitations","MIT-BIH only; reference R-peak positions; beat-level labels; minority-class scarcity; no external or prospective validation; not a medical device.")
    panel("Next upgrades","Cross-database testing • automated R-peak detection • validation-only temperature scaling • ONNX export and latency profiling • rhythm-level aggregation • uncertainty-aware abstention.")
    st.error("Never use this research prototype to make health decisions.")
