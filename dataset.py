import os
from pathlib import Path
import torch
import re
import random
import transformers
from transformers import BertTokenizer
import tqdm
from torch.utils.data import Dataset, DataLoader
import itertools
import math
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torchvision import transforms
import pandas as pd
import json

path = 'polyvore_outfits/disjoint'
chosen_categories = ["tops", "bottoms", "shoes", "jewellery"]

train = json.load(open(f'{path}/train.json'))
val = json.load(open(f'{path}/valid.json'))

metadata = json.load(open(f'{path}/../polyvore_item_metadata.json'))


def filter_ds(ds):
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
                random_outfit = np.random.choice(train)
                negative_item = find_category(random_outfit, last_category, metadata)
                if negative_item:
                    break
            final_outfit.append(negative_item)
            filtered.append(final_outfit)

            # Categoria del 5
            # Outfit randomico
            # Prendere item con la stessa categoria del quinto
    return filtered


def find_category(outfit, category, metadata):
    items = outfit['items']
    for item in items:
        item_cat = metadata[item['item_id']]['semantic_category']
        if item_cat == category:
            item_description = metadata[item['item_id']]['url_name'] + metadata[item['item_id']]['description']
            return {"item": item['item_id'], "description": item_description}
    return None


train_filtered = filter_ds(train)
val_filtered = filter_ds(val)
