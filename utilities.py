from torch import nn
import matplotlib.pyplot as plt
import os
import shutil


# Example list of loss values
# loss_values = [0.037226, 0.033801, 0.032849, 0.032698, 0.032696, 0.033275, 0.031753, 0.031765, 0.031801, 0.032914]
#
# # Plotting the loss values
# plt.plot(loss_values)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss during Epochs')
# plt.show()

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


# Path to the folder containing the images
folder_path = "./polyvore_outfits/images"

# TODO: THIS SCRIPT IS USED TO CREATE SUBFOLDERS STARTING FROM THE FOLDERS COMPOSED OF IMAGES

# Iterate over the images in the folder
# for filename in os.listdir(folder_path):
#     print(f"Image {filename}")
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         # Get the first and last numbers from the filename
#         identifier = filename[:-4]  # Remove the file extension
#         first_number = int(identifier[:1])
#         last_number = int(identifier[-1])
#
#         # Create subfolders based on the first and last numbers
#         subfolder_path = f"./polyvore_outfits/subfolders/{first_number}_{last_number}"
#         os.makedirs(subfolder_path, exist_ok=True)
#
#         # Move the image to the corresponding subfolder
#         source_path = os.path.join(folder_path, filename)
#         destination_path = os.path.join(subfolder_path, filename)
#         shutil.copy(source_path, destination_path)


path = './polyvore_outfits/subfolders'
# model.apply(initialize_weights)
# Path to the parent directory

# Iterate over the directories in the parent directory
num_elements = 0
# TODO:This script is used only to check that no images were lost during the division in subfolders
for directory in os.listdir(path):
    directory_path = os.path.join(path, directory)

    # Check if the current item is a directory
    if os.path.isdir(directory_path):
        num_elements += len(os.listdir(directory_path))
        print(f"Directory: {directory}, Number of Elements: {num_elements}")

print(num_elements)
# Set the directory path on your Google Drive
