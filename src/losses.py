import torch
import torch.nn.functional as F
class FocalLoss(torch.nn.Module):
    def __init__(self,alpha=None,gamma=2):
        super().__init__(); self.register_buffer("alpha",alpha); self.gamma=gamma
    def forward(self,logits,target):
        lp=F.log_softmax(logits,1); logpt=lp.gather(1,target[:,None]).squeeze(1); pt=logpt.exp()
        loss=-(1-pt).pow(self.gamma)*logpt
        if self.alpha is not None: loss=loss*self.alpha[target]
        return loss.mean()
