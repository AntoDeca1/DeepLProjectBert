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

# # ------------Num-of-Images-------------------
# folder_path = "polyvore_outfits/images"
# # Get the list of elements in the folder
# elements = os.listdir(folder_path)
# # Count the number of elements
# num_elements = len(elements)
# print("Number of images in the folder:", num_elements)

# TODO:Questa parte è da togliere da qui.Messa qui solo per prova
train_path = 'polyvore_outfits/disjoint/train.json'
train = json.load(open(train_path))
fill_in_the_blank_train_path = "polyvore_outfits/disjoint/fill_in_blank_train.json"
fill_in_the_blank_train = json.load(open(fill_in_the_blank_train_path))
train_dict = {el['set_id']: el for el in train}  # Mi serve per la parte nuova


# ----------------------------------Parte Nuova----------------------------------------
# Da capire se ha senso seguire questo approccio
# TODO: Aggiustare queste due funzioni perchè prendono dallo scope globale
def from_set_id_to_id(list_of_set_id):
    """
    Used in map_set_id() function.Starting from a list of setid_position return
    a list of item_id
    :param list_of_set_id:
    :return:
    """
    items = []
    for el in list_of_set_id:
        set_id, el_position = el.split("_")
        real_position = int(el_position) - 1
        real_item_id = train_dict[set_id]["items"][real_position]['item_id']
        description = metadata[real_item_id]["description"] + metadata[real_item_id]['url_name']
        category = metadata[real_item_id]["semantic_category"]
        items.append({"item": real_item_id, "description": description, "category": category})
    return items


def map_set_id():
    """
    Starting from fill_in_the_blank_train
    Iterate through each dict.Taking in account only the outfit with 4 items in the chosen categories
    :return:
    """
    filtered_questions = []
    for fill_in_the_blank_dict in fill_in_the_blank_train:
        categories = []
        outfit = []
        if len(fill_in_the_blank_dict["question"]) != 4: continue
        for el in fill_in_the_blank_dict["question"]:
            set_id, el_position = el.split("_")
            real_position = int(el_position) - 1
            real_item_id = train_dict[set_id]["items"][real_position]['item_id']
            category = metadata[real_item_id]["semantic_category"]
            description = metadata[real_item_id]["description"] + metadata[real_item_id]['url_name']
            outfit.append({"item": real_item_id, "description": description, "category": category})
            categories.append(category)
        if all(item in chosen_categories for item in categories):
            answers = from_set_id_to_id(fill_in_the_blank_dict["answers"])
            outfit.extend(answers)
            filtered_questions.append(outfit)
    return filtered_questions


filtered_questions = map_set_id()
print()


# ---------------------------------------------------------------------


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
