import json
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import CLIPProcessor
from typing import Dict, Any, List, Tuple
from PIL import Image
from config import CLIP_MODEL, BENCHMARKS

# Preprocessor
feature_extractor = CLIPProcessor.from_pretrained(CLIP_MODEL)

class BenchmarkDataset(Dataset):
    def __init__(self, name: str):
        """Initializes a benchmark dataset.

        Args:
            dataframe (pd.DataFrame): Augmented dataframe including countries
            version_3k (bool): whether new img2gps dataset should be loaded.
                Note: we need to evaluate on both version of the dataset.
        """
        with open(BENCHMARKS, 'r') as f:
            benchmarks = json.load(f)

        assert name in benchmarks.keys(), f'Benchmark {name} is not a valid benchmark. Specify the \
            metadata and image paths in a JSON file and configure it in config under "BENCHMARKS".'
        
        self.name = name
        self._benchmark = benchmarks[name]
        self.df = pd.read_csv(self._benchmark['meta'])

    def __len__(self) -> int:
        """Returns dataset length.

        Returns:
            int: dataset length
        """
        return len(self.df.index)

    @property
    def panorama(self):
        return False # TODO: Not implemented
    
    @property
    def multi_task(self):
        return False # TODO: Not implemented
    
    def _crop_images(self, image: np.ndarray) -> List[np.ndarray]:
        """Generate crops of the image. Center crop only for now.

        Args:
            image (np.ndarray): numpy array image

        Returns:
            List[np.ndarray]: list of three images.
        """
        s = np.min(image.shape[:2])
    
        # Center
        h_start = int((image.shape[0] - s) / 2)
        w_start = int((image.shape[1] - s) / 2)
        center = image[h_start: h_start+s, w_start:w_start+s]    
        return center

    def _load_jpg(self, filename: str) -> Tuple:
        """Loads image from path.

        Args:
            filename (str): Image name.

        Returns:
            Tuple: Tuple of (PIL image, numpy array).
        """
        pil_image = Image.open(self._benchmark['images'] + filename)
        image = pil_image.convert('RGB')
        image = np.asarray(image, dtype=np.float32) / 255
        image = image[:, :, :3]
        return pil_image, image

    def __getitem__(self, index: Any) -> Dict:
        """Gets sample with index from dataset and preprocesses image.

        Args:
            index (int): sample index

        Returns:
            Dict: dataset sample
        """
        if type(index) == str:
            if index == 'labels':
                return self.df[['lng', 'lat']].values
            
            elif index == 'labels_clf':
                return self.df['geocell_idx_yfcc'].values

            else:
                return self.df[index]

        sample = self.df.iloc[index]
        _, image = self._load_jpg(sample['image'])
        images = self._crop_images(image)
        model_inputs = feature_extractor(images=images, return_tensors='pt')
        model_inputs['labels'] = torch.tensor(sample[['lng', 'lat']])
        model_inputs['labels_clf'] = torch.tensor(sample['geocell_idx_yfcc'])
        return model_inputs