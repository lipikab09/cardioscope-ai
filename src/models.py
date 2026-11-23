import torch
from torch import nn
class ResidualBlock1D(nn.Module):
    def __init__(self,ci,co,stride=1,drop=.2):
        super().__init__()
        self.c1=nn.Conv1d(ci,co,7,stride,padding=3,bias=False); self.b1=nn.BatchNorm1d(co)
        self.c2=nn.Conv1d(co,co,5,padding=2,bias=False); self.b2=nn.BatchNorm1d(co)
        self.act=nn.GELU(); self.drop=nn.Dropout(drop)
        self.skip=nn.Identity() if ci==co and stride==1 else nn.Sequential(nn.Conv1d(ci,co,1,stride,bias=False),nn.BatchNorm1d(co))
    def forward(self,x):
        r=self.skip(x); x=self.act(self.b1(self.c1(x))); x=self.drop(self.b2(self.c2(x)))
        return self.act(x+r)
class CardioScopeNet(nn.Module):
    def __init__(self,n_classes=5,dropout=.25):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(1,32,9,padding=4,bias=False),nn.BatchNorm1d(32),nn.GELU())
        self.block1=ResidualBlock1D(32,48,2,dropout); self.block2=ResidualBlock1D(48,80,2,dropout); self.block3=ResidualBlock1D(80,128,2,dropout)
        self.attention=nn.Sequential(nn.Conv1d(128,1,1),nn.Softmax(dim=-1))
        self.head=nn.Sequential(nn.Linear(256,96),nn.GELU(),nn.Dropout(dropout),nn.Linear(96,n_classes))
    def forward(self,x):
        x=self.stem(x); x=self.block1(x); x=self.block2(x); f=self.block3(x)
        a=self.attention(f); pooled=torch.cat([f.mean(-1),(f*a).sum(-1)],1)
        return self.head(pooled)
