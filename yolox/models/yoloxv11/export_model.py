from darknet_pd import Darknet
import torch

config = './darknet_v1_anchor_free.cfg'

model = Darknet(config, 'cpu')
model.eval()
x = torch.randn(1, 3, 384, 384)
y = model(x)
for item in y:
    print(item.shape)
print(model)
