import logging
import numpy as np
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import DatasetDict
from dataset_creation.finetune import EmbedDataset
from transformers import AutoModel 
from tqdm.auto import tqdm
from typing import Any
from config import EMBED_BATCH_SIZE_PER_GPU

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('embed')

def compute_embeddings(name: str, model: AutoModel, data: DataLoader, 
                       accelerator: Accelerator) -> np.ndarray:
    """Computes all embeddings of a given dataset split in parallel across multiple GPUs.

    Args:
        name (str): Split name.
        model (AutoModel): Model to use for creating embeddings.
        data (DataLoader): Data Loader.
        accelerator (Accelerator): Multi-GPU accelerator.

    Returns:
        np.ndarray: Numpy array of embeddings.
    """
    logger.warning(f'Starting {name} embedding ...')
    all_outputs = []
    all_indices = []
    for _, (pixels, index) in tqdm(enumerate(tqdm(data)), disable=not accelerator.is_local_main_process):
        output = model(pixels)

        # Gather all predictions and targets
        all_indic = accelerator.gather(index)
        all_output = accelerator.gather(output)
        all_outputs.append(all_output.cpu().detach().numpy())
        all_indices.append(all_indic.cpu().detach().numpy())

    if accelerator.is_local_main_process:
        np.save(f'data/landmark_embeddings/{name}.npy', all_outputs)
        np.save(f'data/landmark_embeddings/{name}_indices.npy', all_indices)

def embed_images(loaded_model: Any, dataset: DatasetDict):
    """Embedding images with multi-GPU support.

    Args:
        loaded_model (Any):     Model used for embedding images.
        dataset (DatasetDict):  Dataset containing train, val, and test splits.

    Raises:
        Exception: If all embeddings have been successfully computed and saved to file.
    """
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False) #find_unused_parameters=True
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Cast to embed dataset
    for split_name, split_dataset in dataset.items():
        dataset[split_name] = EmbedDataset(split_dataset)
    
    # Create dataloaders
    train_data = DataLoader(dataset['train'], EMBED_BATCH_SIZE_PER_GPU, shuffle=False, num_workers=8)
    val_data = DataLoader(dataset['val'], EMBED_BATCH_SIZE_PER_GPU, shuffle=False, num_workers=8)
    test_data = DataLoader(dataset['test'], EMBED_BATCH_SIZE_PER_GPU, shuffle=False, num_workers=8)

    # Data loader
    model, train_data, val_data, test_data = accelerator.prepare(loaded_model, train_data, val_data, test_data)

    # Set to train mode
    model.eval()

    # Compute embeddings
    compute_embeddings('train', model, train_data, accelerator)
    accelerator.wait_for_everyone()
    compute_embeddings('val', model, val_data, accelerator)
    accelerator.wait_for_everyone()
    compute_embeddings('test', model, test_data, accelerator)
    accelerator.wait_for_everyone()

    # Cast back to normal dataset
    for split_name, split_dataset in dataset.items():
        dataset[split_name] = split_dataset.dataset