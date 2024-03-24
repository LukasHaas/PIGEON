import json
import random
import torch
import numpy as np
import pandas as pd
from random import shuffle
from PIL import Image
from typing import Tuple, Any
from datasets import DatasetDict
from transformers import CLIPProcessor
from torchvision.transforms import transforms
from config import CLIP_MODEL, IMAGE_PATH_YFCC, PRETRAIN_METADATA_PATH_YFCC, DRIVING_SIDE_PATH

# Initialize CLIP image processor
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

# Driving Side
with open(DRIVING_SIDE_PATH) as f:
    DRIVING_SIDE = json.load(f)

# Renaming
RENAMING = {
 'South Georgia and the South Sand': 'South Georgia and the South Sandwich Islands',
 'United States Minor Outlying Isl': 'United States Minor Outlying Islands'
}

# Plural countries
PLURALS = [
 'Bahamas',
 'British Virgin Islands',
 'Caspian Sea',
 'Cayman Islands',
 'Central African Republic',
 'Cocos Islands',
 'Cook Islands',
 'Democratic Republic of the Congo',
 'Dominican Republic',
 'Falkland Islands',
 'Faroe Islands',
 'Isle of Man',
 'Maldives',
 'Netherlands',
 'Northern Mariana Islands',
 'Philippines',
 'Republic of the Congo',
 'Seychelles',
 'Solomon Islands',
 'Turks and Caicos Islands',
 'United Arab Emirates',
 'United Kingdom',
 'United States',
 'United States Minor Outlying Isl',
 'Vatican City',
 'Virgin Islands, U.S.',
 'Western Sahara',
]

class PretrainDatasetYFCC(torch.utils.data.Dataset):
    """Dataset used for pretraining CLIP with geolocalization data from YFCC.
    """
    def __init__(self, split: str, metadata: str=PRETRAIN_METADATA_PATH_YFCC, auxiliary: bool=True,
                 shuffle: bool=True, image_size: int=336):
        """Initializes a PretrainDatasetYFCC used for pretraining CLIP.

        Args:
            split (str): dataset split to load.
            metadata (str, optional): path to metadata csv file containing image filepaths
                and auxiliary information. Defaults to PRETRAIN_METADATA_PATH_YFCC.
            auxiliary (bool, optional): whether to use auxiliary information for pretraining.
                Defaults to True.
            shuffle (bool, optional): whether the training data should be shuffled.
                Defaults to True.
            image_size (int, optional): the size to which the image should be resized.
        """
        assert split in ['train', 'val', 'test']
        self.df = pd.read_csv(metadata).query(f"selection == '{split}'")
        if shuffle:
            self.df = self.df.sample(frac=1.0, random_state=330)

        self.df = self.df.reset_index(drop=True)
        self.auxiliary = auxiliary
        self.shuffled = shuffle
        self.image_size = image_size

        no_str = 'no ' if auxiliary == False else ''
        shuffle_str = 'shuffled ' if shuffle else ''
        print(f'Initialized {shuffle_str}{split} YFCC dataset with {no_str}auxiliary data.')

    def _is_valid(self, value: Any):
        """Checks whether the provided value is valid.

        Args:
            value (Any): any value.
        """
        return type(value) == str or np.isnan(value) == False
    
    def _select_caption(self, index: int) -> str:
        """Generates a random caption for the given image using auxiliary data.

        Args:
            index (int): row index to generate caption for.

        Returns:
            str: randomly generated caption.
        """
        s = self.df.iloc[index]

        # Country, region, town
        country = s.country_name
        if country == 'United States Of America':
            country = 'United States'

        # Slight renaming
        plural = country in PLURALS
        display_country = country if country not in RENAMING else RENAMING[country]
        display_country = display_country if not plural else f'the {display_country}'

        if self._is_valid(s.geo_area) and random.random() > 0.0: # Always show area and town when available
            region_string = f'in the region of {s.geo_area} '
        else:
            region_string = ''

        if self._is_valid(s.town) and random.random() > 0.0: # Always show area and town when available
            town_string = f'close to the town of {s.town} '
        else:
            town_string = ''

        # Climate zone
        if self._is_valid(s.climate_zone) and random.random() > 0.55:
            climate_caption = f' This location has a {s.climate_zone.lower()} climate.'
        else:
            climate_caption = ''

        # Location
        if random.random() > 0.2 or climate_caption == '' or self.auxiliary == False:
            location_caption = f'A photo I took {town_string}{region_string}in {display_country}.'
            if self.auxiliary == False:
                return location_caption
        else:
            location_caption = ''
        
        # Driving right or left
        if country in DRIVING_SIDE.keys() and climate_caption == '' and random.random() > 0.8:
            driving_right_caption = f' In this location, people drive on the {DRIVING_SIDE[country]} side of the road.'
        else:
            driving_right_caption = ''
            
        other_components = [climate_caption, driving_right_caption]
        shuffle(other_components)
        components = [location_caption] + other_components
        caption = ''.join(components).strip()
        return caption

    def _crop_resize(self, image: Image.Image) -> Image.Image:
        """Crops and resizes the given image.
        
        Args:
            image (Image.Image): The image to be cropped and resized.
            
        Returns:
            Image.Image: The cropped and resized image.
        """
        # Crop the image to the largest possible square
        width, height = image.size
        new_dim = min(width, height)
        left = (width - new_dim) / 2
        top = (height - new_dim) / 2
        right = (width + new_dim) / 2
        bottom = (height + new_dim) / 2
        image = image.crop((left, top, right, bottom))

        # Resize the cropped image to a side length of self.image_size pixels
        return image #.resize((self.image_size, self.image_size))

    def __getitem__(self, index: int) -> Tuple:
        """Retrieves item in dataset for given index.

        Args:
            index (int): sample index.

        Returns:
            Dict: sample model input
        """
        #print('Retrieving YFCC index:', index)

        # Load the image
        image_filename = self.df.iloc[index]['image']
        image = Image.open(IMAGE_PATH_YFCC + '/' + image_filename)

        # Crop image
        image = self._crop_resize(image)

        # Generate a random caption for the image
        caption = self._select_caption(index)
        return image, caption
    
    def __len__(self):
        return len(self.df.index)

    @classmethod
    def generate(cls, metadata: str=PRETRAIN_METADATA_PATH_YFCC, auxiliary: bool=True) -> DatasetDict:
        """Generates a DatasetDict with PretrainedDatasets.

        Args:
            split (str): dataset split to load.
            metadata (str, optional): path to metadata csv file containing image filepaths
                and auxiliary information. Defaults to METADATA_PATH.
            auxiliary (bool, optional): whether to use auxiliary information for pretraining.
                Defaults to True.

        Returns:
            DatasetDict: dataset dictionary from train, val, and test.
        """
        return DatasetDict(
            train=cls('train', metadata, auxiliary),
            val=cls('val', metadata, auxiliary),
            test=cls('test', metadata, auxiliary)
        )

    def accuracy(self, model: Any, batch_size: int, trials: int=30) -> float:
        """Computes the accuracy of a given mode on the current dataset.

        Args:
            model (Any): pretrained CLIP model.
            batch_size (int): batch size of model
            trials (int, optional): Number of runs for the Monte-Carlo estimation
                of accuracy. Defaults to 30.

        Returns:
            float: accuracys
        """
        accs = []
        for t in range(trials):
            inputs = [self[(t * batch_size) + i] for i in range(batch_size)]
            images, captions = zip(*inputs)
            images = list(images)
            captions = list(captions)

            inputs = clip_processor(text=captions, images=images, return_tensors='pt',
                                    padding=True, truncation=True)
            for key in inputs:
                inputs[key] = inputs[key].to('cuda')

            inputs['return_loss'] = True
            outputs = model(**inputs)
            predictions = outputs.logits_per_image.softmax(dim=1).argmax(dim=1)
            accuracy = (predictions == torch.arange(batch_size, device='cuda')).sum()
            accs.append(accuracy / batch_size)
        
        acc = sum(accs) / trials
        return acc