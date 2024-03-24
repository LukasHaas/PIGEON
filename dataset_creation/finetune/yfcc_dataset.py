import os
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

class YFCCDataset:
    def __init__(self, dataset: Dataset, split: str):
        """
        A thin wrapper around a YFCC dataset which loads embeddings from disk.

        Args:
            dataset (Any): Dataset.
        """
        self.dataset = dataset
        self.split = split

        split_files = [x for x in os.listdir('data/yfcc_embeddings') if 'npz' in x]
        split_files = [x for x in split_files if self.split in x]
        self._embedding_dicts = []
        for file in split_files:
            print('... loading file:', file)
            dict_ = np.load(f'data/yfcc_embeddings/{file}')
            self._embedding_dicts.append(dict_)

    def _search_for_embedding(self, index: int) -> Tensor:
        """Retrieves embeddings from file.

        Args:
            index (int): Index to retrieve embedding for.

        Returns:
            Tensor: Embedding.
        """
        idx = index.item()
        for i, dict_ in enumerate(self._embedding_dicts):
            if str(idx) in dict_:
                embedding = torch.from_numpy(dict_[str(idx)])
                return {
                    'embedding': embedding
                }
        
        raise Exception(f'Index {idx} not present.')

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data['embedding'] = self._search_for_embedding(data['index'])
        del data['index']
        return data

    def __len__(self):
        return len(self.dataset)