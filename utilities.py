from torch import nn
import matplotlib.pyplot as plt

# Example list of loss values
loss_values = [0.037226, 0.033801, 0.032849, 0.032698, 0.032696, 0.033275, 0.031753, 0.031765, 0.031801, 0.032914]

# Plotting the loss values
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss during Epochs')
plt.show()


def count_parameters(model):
    """
    Count the number of parameters of a pytorch model
    :param model:
    :return:
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    """
    How to inizialize weights with Xavier.
    :param m:
    :return:
    """
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

# model.apply(initialize_weights)
