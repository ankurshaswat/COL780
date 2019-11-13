import torch
a = torch.rand(20000,20000).cuda()
while True:
    a += 1
    a -= 1