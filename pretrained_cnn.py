# fetch pretrained model
import torch
import ssl
from dataset import loader
import numpy

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
layer = model._modules.get('avgpool')
print(layer)


def copy_embeddings(m, i, o):
    """Copy embeddings from the penultimate layer.
    """
    o = o[:, :, 0, 0].detach().numpy().tolist()
    outputs.append(o)


outputs = []
# attach hook to the penulimate layer
_ = layer.register_forward_hook(copy_embeddings)
model.eval()  # Inference mode
# Generate image's embeddings for all images in dloader and saves
# them in the list outputs
for X in loader:
    sequence_len = len(X)
    for i, batch_element in enumerate(X):
            embeddings = model(batch_element)


print(len(outputs))  # returns 92
