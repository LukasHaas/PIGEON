from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from config import CLIP_MODEL

class EmbedDataset:
    def __init__(self, dataset: Dataset):
        """
        A thin wrapper around a dataset which loads images from disk.

        Args:
            dataset (Any): Dataset.
        """
        self.dataset = dataset
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        pil_image = data['image']
        inputs = self.processor(images=pil_image, return_tensors='pt')
        pixel_values = inputs['pixel_values']
        return pixel_values.squeeze(), data['index']

    def __len__(self):
        return len(self.dataset)