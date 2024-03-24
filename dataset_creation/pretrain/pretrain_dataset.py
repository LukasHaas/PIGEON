import os
import random
import torch
import numpy as np
import pandas as pd
from random import shuffle
from PIL import Image
from typing import Tuple, Any
from datasets import DatasetDict, Dataset
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import RandomCrop, CenterCrop
from config import CLIP_MODEL, IMAGE_PATH, IMAGE_PATH_2, PRETRAIN_METADATA_PATH

# Initialize CLIP image processor
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

# Initialize Cropper for second batch of streetview images
l_cropper = CenterCrop(192.53436408909982 * 2)
v_cropper = CenterCrop(596)

MONTHS = {
    0: 'January',
    1: 'February',
    2: 'March',
    3: 'April',
    4: 'May',
    5: 'June',
    6: 'July',
    7: 'August',
    8: 'September',
    9: 'October',
    10: 'November',
    11: 'December'
}

THE_LIST = ['Bahamas',
 'British Virgin Islands',
 'Cayman Islands',
 'Cocos Islands',
 'Comoros',
 'Cook Islands',
 'Falkland Islands',
 'Faroe Islands',
 'French Southern Territories',
 'Maldives',
 'Marshall Islands',
 'Netherlands',
 'Northern Mariana Islands',
 'Paracel Islands',
 'Philippines',
 'Pitcairn Islands',
 'Seychelles',
 'Solomon Islands',
 'Spratly Islands',
 'Turks and Caicos Islands',
 'United Arab Emirates',
 'United States']

class PretrainDataset(torch.utils.data.Dataset):
    """Dataset used for pretraining CLIP with geolocalization data.
    """
    def __init__(self, split: str, metadata: str=PRETRAIN_METADATA_PATH, auxiliary: bool=True):
        """Initializes a PretrainDataset used for pretraining CLIP.

        Args:
            split (str): dataset split to load.
            metadata (str, optional): path to metadata csv file containing image filepaths
                and auxiliary information. Defaults to METADATA_PATH.
            auxiliary (bool, optional): whether to use auxiliary information for pretraining.
                Defaults to True.
        """
        assert split in ['train', 'val', 'test']
        self.df = pd.read_csv(metadata).query(f"selection == '{split}'")
        self.df = self.df.reset_index(drop=True)
        self.auxiliary = auxiliary

        # Dataset cutoffs
        self.cutoff_1 = len(self.df[self.df['source'].str.startswith('o')].index) * 4
        self.cutoff_2 = len(self.df[self.df['source'].str.startswith('v')].index) + self.cutoff_1
        self.cutoff_3 = len(self.df[self.df['source'].str.startswith('l')].index) * 5 + self.cutoff_2

        no_str = 'no' if auxiliary == False else ''
        print(f'Initialized {split} dataset with {no_str} auxiliary data.')

    def _convert_to_row_index(self, index: int) -> Tuple:
        """Converts the dataset index to the row index in the dataframe

        Args:
            index (int): dataset index

        Returns:
            Tuple: (row index, image index)
        """
        if index < self.cutoff_1:
            row_index = int(index / 4)
            image_col = index % 4

        elif index < self.cutoff_2:
            row_index = int(self.cutoff_1 / 4) + (index - self.cutoff_1)
            image_col = 0

        else:
            row_index =  int(self.cutoff_1 / 4) + (self.cutoff_2 - self.cutoff_1) \
                         + int((index - self.cutoff_2) / 5)
            image_col = (index - self.cutoff_2) % 5
        
        return row_index, image_col
    
    def _select_image(self, index: int, image_col_idx: int) -> Tuple:
        """Selects one random image for the given sample.

        Args:
            index (int): row index to retrieve.
            image_col_idx (int): number of image to retrieve.

        Returns:
            Tuple: (Image, heading offset)
        """
        s = self.df.iloc[index]
        if s.source.startswith('o'):
            # Select the correct of the four images for the given datapoint
            image_col = [x for x in self.df.columns if 'image' in x][image_col_idx]
            image_filename = s[image_col]

            # Load the image
            image = Image.open(IMAGE_PATH + '/' + image_filename)

            # Calculate the heading offset
            angle_offset = image_col_idx * 90

        elif s.source.startswith('l'):
            # Randomly select one of the 5 images for the given datapoint
            img_start = image_col_idx * 512
            img_end = (image_col_idx + 1) * 512

            # Load the image
            image_filename = s['image']
            image = Image.open(IMAGE_PATH_2 + '/' + image_filename)
            image = image.convert('RGB')
            image = np.asarray(image, dtype=np.float32) / 255

            # Select and crop image
            image = image[:, img_start:img_end, :3] * 255
            image = image.astype(np.uint8)
            image = Image.fromarray(image)
            image = l_cropper(image)

            # Calculate the heading offset
            angle_offset = image_col_idx * 72

        elif s.source.startswith('v'):
            # Load the image
            image_filename = s['image']
            image = Image.open(image_filename)
            image = v_cropper(image)
            angle_offset = 0

        else:
            raise Exception(f'Invalid image source: {s.source}')

        return image, angle_offset

    def _is_valid(self, value: Any):
        """Checks whether the provided value is valid.

        Args:
            value (Any): any value.
        """
        return type(value) == str or np.isnan(value) == False
    
    def _select_caption(self, index: int, heading_offset: int) -> str:
        """Generates a random caption for the given image using auxiliary data.

        Args:
            index (int): row index to generate caption for.
            heading_offset (int): heading offset in angles from north.

        Returns:
            str: randomly generated caption.
        """
        s = self.df.iloc[index]
        # Country, region, town
        country = s.country_name
        if country == 'United States Of America':
            country = 'United States'

        country = country if country not in THE_LIST else f'the {country}'

        if self._is_valid(s.geo_area) and random.random() > 0.4:
            region_string = f'in the region of {s.geo_area} '
        else:
            region_string = ''

        if self._is_valid(s.town) and random.random() > 0.6:
            town_string = f'close to the town of {s.town} '
        else:
            town_string = ''

        # Climate zone
        if self._is_valid(s.climate_zone) and random.random() > 0.6:
            climate_caption = f' This location has {s.climate_zone.lower()}.'
        else:
            climate_caption = ''

        # Location
        if random.random() > 0.3 or climate_caption == '' or self.auxiliary == False:
            location_caption = f'A Street View photo {town_string}{region_string}in {country}.'
            if self.auxiliary == False:
                return location_caption
        else:
            location_caption = ''
        
        # Driving right or left
        if self._is_valid(s.driving_right) and climate_caption == '' and random.random() > 0.7:
            direction = 'right' if s.driving_right else 'left'
            driving_right_caption = f' In this location, people drive on the {direction} side of the road.'
        else:
            driving_right_caption = ''
            
        # Compass direction
        if self._is_valid(s.heading) and random.random() > 0.7:
            compass_direction = (s.heading + heading_offset) % 360
            if compass_direction <= 45 or compass_direction > 315:
                compass_caption = ' This photo is facing north.'
            elif compass_direction > 45 and compass_direction <= 135:
                compass_caption = ' This photo is facing east.'
            elif compass_direction > 135 and compass_direction <= 225:
                compass_caption = ' This photo is facing south.'
            elif compass_direction > 225 and compass_direction <= 315:
                compass_caption = ' This photo is facing west.'
        else:
            compass_caption = ''
                
        # Month (because of seasons)
        if self._is_valid(s.month) and random.random() > 0.7:
            month_caption = f" The photo was taken in {MONTHS[s.month]}."
        else:
            month_caption = ""
        
        other_components = [climate_caption, driving_right_caption,compass_caption, month_caption]
        shuffle(other_components)
        components = [location_caption] + other_components
        caption = ''.join(components).strip()
        return caption

    def _random_transform(self, image: Image) -> Image:
        """Randomly transforms the image on data load.

        Args:
            image (Image): image.

        Returns:
            Image: transformed image.
        """
        side_length, _ = image.size
        cropped_length = random.uniform(0.8, 1) * side_length
        cropper = RandomCrop(cropped_length)
        return cropper(image)

    def __getitem__(self, index: int) -> Tuple:
        """Retrieves item in dataset for given index.

        Args:
            index (int): sample index.

        Returns:
            Dict: sample model input
        """
        row_index, image_col = self._convert_to_row_index(index)

        # Randomly select one of the four images
        image, heading_offset = self._select_image(row_index, image_col)
        caption = self._select_caption(row_index, heading_offset)
        return image, caption
    
    def __len__(self):
        return self.cutoff_3

    @classmethod
    def generate(cls, metadata: str=PRETRAIN_METADATA_PATH, auxiliary: bool=True) -> DatasetDict:
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
            images, captions, _ = zip(*inputs)
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