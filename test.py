import torch
import torch.nn.functional as F

a = torch.Tensor([[1,1,1,1],[1,2,3,4]])
b = F.normalize(a, dim=1)
print(b)
c = "测试测试"
print(c[:2])