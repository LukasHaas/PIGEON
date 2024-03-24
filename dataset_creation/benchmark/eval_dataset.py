import torch
from datasets import Dataset, DatasetDict
from transformers import CLIPProcessor
from preprocessing import extract_features
from config import CLIP_MODEL
from typing import Dict, Any
from PIL import Image

# Initialize CLIP image processor
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

class EvalDataset(Dataset):
    def __init__(self, data: Dataset):
        """Initializes a streetview dataset.

        Args:
            data (Dataset): dataset with image paths
            pretrain (bool, optional): whether the dataset is used for
                pretraining CLIP. Defaults to False.
        """
        self.streetview_data = data

    def __len__(self) -> int:
        """Returns dataset length.

        Returns:
            int: dataset length
        """
        return len(self.streetview_data)

    def _get_multiple_features(self, samples: Dict) -> Dict:
        """Extracts features for multiple samples at the same time.

        Args:
            samples (Dict): selection of samples.

        Returns:
            Dict: transformed dictionary.
        """
        x = {}
        images = [Image.fromarray(x.numpy()) for x in samples['image']]
        x['pixel_values'] = clip_processor(images=images, return_tensors='pt')['pixel_values']
        if 'image_2' in samples:
            for i in range(2, 5):
                images = [Image.fromarray(x.numpy()) for x in samples[f'image_{i}']]
                x[f'pixel_values_{i}'] = clip_processor(images=images,
                                                        return_tensors='pt')['pixel_values']

            x['pixel_values'] = torch.concat((x['pixel_values'], x['pixel_values_2'], x['pixel_values_3'],
                                            x['pixel_values_4']), dim=1)

        return x['pixel_values']

    def __getitem__(self, index: Any) -> Dict:
        """Gets sample with index from dataset and preprocesses image

        Args:
            index (int): sample index

        Returns:
            Dict: dataset sample
        """
        if type(index) == int:
            sample = self.streetview_data[index]
            sample['pixel_values'] = extract_features(sample, clip_processor)['pixel_values']
            sample = {key: value for key, value in sample.items() if 'image' not in key}
            return sample

        elif type(index) == slice:
            samples = self.streetview_data[index]
            samples['pixel_values'] = self._get_multiple_features(samples)
            samples = {key: value for key, value in samples.items() if 'image' not in key}
            return samples

        return self.streetview_data[index]

    @classmethod
    def transform(cls, dataset: DatasetDict) -> DatasetDict:
        """Transforms a dataset dict into a EvalDataset dict

        Args:
            dataset (DatasetDict): original dataset dict

        Returns:
            DatasetDict: transformed dataset dict
        """
        if type(dataset) == DatasetDict:
            return DatasetDict(
                train=cls(dataset['train']),
                val=cls(dataset['val']),
                test=cls(dataset['test'])
            )

        return cls(dataset)