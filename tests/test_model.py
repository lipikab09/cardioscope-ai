import torch
from src.models import CardioScopeNet
def test_shape(): assert CardioScopeNet()(torch.randn(3,1,252)).shape==(3,5)
