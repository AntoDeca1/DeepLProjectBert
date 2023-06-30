import torch
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image
import json
import os


chosen_categories = ["tops", "bottoms", "shoes", "jewellery"]
metadata = json.load(open("polyvore_outfits/polyvore_item_metadata.json"))
# ------------Num-of-Images-------------------
folder_path = "polyvore_outfits/images"
# Get the list of elements in the folder
elements = os.listdir(folder_path)
# Count the number of elements
num_elements = len(elements)
print("Number of images in the folder:", num_elements)


# -------------------------------------------

class CustomDataset(Dataset):
    OPTIMAL_LENGTH_PERCENTILE = 70

    def __init__(self, path):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.cnn_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.path = path
        self.raw_dataset = json.load(open(path))
        self.filtered_dataset = self.filter_ds(self.raw_dataset)
        self.optimal_length = self.compute_optimal_length()
        print()

    def filter_ds(self, ds):
        """
        Iterate through the training set and select only the outfit of 5 elements with the given categories
        :param ds:
        :return:
        """
        filtered = []
        for outfit in ds:
            if len(outfit['items']) != 5: continue
            final_outfit = [None, None, None, None, None]
            for item in outfit['items']:
                item_semantic_cat = metadata[item['item_id']]['semantic_category']
                item_description = metadata[item['item_id']]['url_name'] + metadata[item['item_id']]['description']

                if item_semantic_cat not in chosen_categories:
                    item_cat_pos = 4
                else:
                    item_cat_pos = chosen_categories.index(item_semantic_cat)
                final_outfit[item_cat_pos] = {"item": item['item_id'], "description": item_description}
            if None not in final_outfit:
                last_category = metadata[final_outfit[-1]['item']]['semantic_category']
                while True:
                    random_outfit = np.random.choice(ds)
                    negative_item = self.find_category(random_outfit, last_category, metadata)
                    if negative_item:
                        break
                final_outfit.append(negative_item)
                filtered.append(final_outfit)
        return filtered

    def find_category(self, outfit, category, metadata):
        """
        Given an outfit find the outfit(if exist) with the given category
        :param outfit:
        :param category:
        :param metadata:
        :return:
        """
        items = outfit['items']
        for item in items:
            item_cat = metadata[item['item_id']]['semantic_category']
            if item_cat == category:
                item_description = metadata[item['item_id']]['url_name'] + metadata[item['item_id']]['description']
                return {"item": item['item_id'], "description": item_description}
        return None

    def __len__(self):
        return len(self.filtered_dataset)

    def get_image(self, image_path):
        """
        Retrieve an image given a path
        :param image_path:
        :return: image(PIL)
        """
        image = Image.open(image_path)
        return image

    def create_image_path(self, id):
        """
        Create the image path
        :param id: Image Id
        :return: The path where the image is
        """
        return "./polyvore_outfits/images/" + id + ".jpg"

    def __getitem__(self, idx):
        images = []
        descriptions = []
        outfit = self.filtered_dataset[idx]
        for item_dict in outfit:
            item_id = item_dict['item']
            item_description = item_dict['description']
            image_path = self.create_image_path(item_id)
            image = self.get_image(image_path)
            image_tensor = self.cnn_preprocess(image)
            tokenized_description = \
                self.tokenizer(item_description, truncation=True, padding='max_length', max_length=self.optimal_length)[
                    'input_ids']
            images.append(image_tensor)
            descriptions.append(tokenized_description)

        images_tensor = torch.stack(images)
        descriptions_tensor = torch.tensor(descriptions)
        return images_tensor, descriptions_tensor

    def compute_optimal_length(self):
        """
        Iterate through the whole training set and after saving all description lengths
        compute the optimal length
        "Description bigger than optimal length will be truncated instead the smaller ones
        padded
        :return:optimal_length
        """
        lenghts = []
        for outfit in self.filtered_dataset:
            for item in outfit:
                item_description = item['description']
                tokenized_description = self.tokenizer(item_description)['input_ids']
                lenghts.append(len(tokenized_description))
        return int(np.percentile(lenghts, self.OPTIMAL_LENGTH_PERCENTILE))
