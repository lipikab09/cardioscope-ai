import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,classification_report
def classification_metrics(y,p,classes):
    pred=p.argmax(1)
    return {"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
      "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
      "weighted_f1":float(f1_score(y,pred,average="weighted",zero_division=0)),
      "confusion_matrix":confusion_matrix(y,pred,labels=np.arange(len(classes))).tolist(),
      "report":classification_report(y,pred,labels=np.arange(len(classes)),target_names=classes,output_dict=True,zero_division=0)}
def patient_bootstrap_macro_f1(y,pred,records,n_boot=1000,seed=42):
    rng=np.random.default_rng(seed); unique=np.unique(records); vals=[]
    for _ in range(n_boot):
        sample=rng.choice(unique,len(unique),replace=True)
        ix=np.concatenate([np.flatnonzero(records==r) for r in sample])
        vals.append(f1_score(y[ix],pred[ix],average="macro",zero_division=0))
    return [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))]
