import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.profiler import profile
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import DatasetDict, Dataset
from transformers import TrainingArguments, AutoModel 
from typing import Any, Callable
from tqdm.auto import tqdm
from dataset_creation.benchmark import BenchmarkDataset
from models import ProtoRefiner
from config import CURRENT_SAVE_PATH

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train')

def generate_profiler() -> profile:
    """Profile to identify bottlenecks

    Returns:
        profile: PyTorch profiler
    """
    return profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=10, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('runs/profile'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )

def evaluate_model(model: nn.Module, dataset: Dataset, metrics: Callable,
                   train_args: TrainingArguments, refiner: ProtoRefiner=None,
                   yfcc: bool=False, writer: SummaryWriter=None, step: int=0) -> float:
    """Evaluates model on evaluation data

    Args:
        model (nn.Module): model to use for evaluation
        dataset (Dataset): validation dataset
        metrics (Callable): function returning a dict of metrics given predictions
                            and labels
        train_args (TrainingArguments): training arguments
        refiner (ProtoRefiner, optional): guess refinement model. Defaults to None.
        yfcc (bool, optional): whether yfcc input data was used.
        writer (SummaryWriter, optional): TensorBoard writer. Defaults to None.
        step (int, optional): number of evaluation step. Defaults to 0.

    Returns:
        float: evaluation loss
    """
    logger.warning(f'Starting evaluation ...')
    eval_data = DataLoader(dataset, train_args.per_device_eval_batch_size, shuffle=False,
                           pin_memory=False, num_workers=16)

    if writer is None:
        writer = SummaryWriter()

    # Set to eval
    model.eval()
    if refiner is not None:
        refiner.eval()

    # Get predictions
    with torch.no_grad():
        combined_loss, combined_loss_clf, combined_loss_reg, combined_loss_climate, combined_loss_month = 0, 0, 0, 0, 0
        combined_preds = []
        combined_geocell_preds = []
        combined_preds_mt = []
        combined_preds_climate = []
        combined_preds_month = []
        combined_top5_cells = []
        combined_top5_probs = []

        for data in tqdm(eval_data):

            # Predict geocell
            outputs = model(**data)
            combined_loss += outputs.loss * len(data)
            combined_loss_clf += outputs.loss_clf * len(data)

            # Geo-regression
            preds_mt = None
            if outputs.loss_reg is not None and outputs.loss_reg > 0:
                combined_loss_reg += outputs.loss_reg * len(data)
                combined_loss_climate += outputs.loss_climate * len(data)
                combined_loss_month += outputs.loss_month * len(data)

                combined_preds_mt.append(outputs.preds_mt.cpu().detach().numpy())
                combined_preds_climate.append(outputs.preds_climate.cpu().detach().numpy())

                if not yfcc:
                    combined_preds_month.append(outputs.preds_month.cpu().detach().numpy())

            # Refine Guess
            if refiner is not None:
                _, preds_LLH, _ = refiner(outputs.embedding,
                                          initial_preds=outputs.preds_LLH,
                                          candidate_cells=outputs.top5_geocells.indices,
                                          candidate_probs=outputs.top5_geocells.values)
                combined_preds.append(preds_LLH.cpu().detach().numpy())
            else:
                combined_preds.append(outputs.preds_LLH.cpu().detach().numpy())

            # Collect data
            if outputs.preds_geocell is not None:
                combined_geocell_preds.append(outputs.preds_geocell.cpu().detach().numpy())
                top5 = outputs.top5_geocells
                combined_top5_cells.append(top5.indices.cpu().detach().numpy())
                combined_top5_probs.append(top5.values.cpu().detach().numpy())

        # Labels
        labels_lla = dataset['labels']
        labels_cell = dataset['labels_clf']
        if isinstance(labels_lla, np.ndarray) == False:
            labels_lla = labels_lla.numpy()
            labels_cell = labels_cell.numpy()

        # Combine predictions
        preds = np.concatenate(combined_preds, axis=0)
        preds_geocells = np.concatenate(combined_geocell_preds, axis=0)
        top5_geocells = np.concatenate(combined_top5_cells, axis=0)

        # Multi-task
        labels_mt = dataset['labels_multi_task'] if combined_loss_reg > 0 else None
        labels_climate = dataset['labels_climate'] if combined_loss_climate > 0 else None
        labels_month = dataset['labels_month'] if combined_loss_month > 0 else None
        preds_mt = np.concatenate(combined_preds_mt, axis=0) if combined_loss_reg > 0 else None
        preds_climate = np.concatenate(combined_preds_climate, axis=0) if combined_loss_climate > 0 else None

        preds_month = None
        if not yfcc:
            preds_month = np.concatenate(combined_preds_month, axis=0) if combined_loss_month > 0 else None

        # Compute metrics
        results = (preds, preds_geocells, preds_mt, preds_climate, preds_month, top5_geocells, \
                  labels_lla, labels_cell, labels_mt, labels_climate, labels_month)
        eval_dict = metrics(results)


    # Write loss to TensorBoard
    if isinstance(dataset, BenchmarkDataset) == False:
        writer.add_scalar('Loss/val', combined_loss / len(dataset), step)
        writer.add_scalar('Loss_clf/val', combined_loss_clf / len(dataset), step)

        if outputs.loss_reg is not None and outputs.loss_reg > 0:
            writer.add_scalar('Loss_reg/val', combined_loss_reg / len(dataset), step)
            writer.add_scalar('Loss_climate/val', combined_loss_climate / len(dataset), step)
            writer.add_scalar('Loss_month/val', combined_loss_month / len(dataset), step)

        # Write metrics to TensorBoard
        for metric, value in eval_dict.items():
            writer.add_scalar(metric, value, step)
    
    model.train()
    logger.warning(f'Back to training ...')

    # Return classification loss to save best classification model
    return -eval_dict['Geocell_accuracy']


def train_model(loaded_model: Any, dataset: DatasetDict, on_embeddings: bool, yfcc: bool, 
                train_args: TrainingArguments, metrics: Callable, patience: int=None,
                should_profile: bool=True) -> AutoModel:
    """Training and evaluation loop for the model with multi-GPU support.

    Args:
        loaded_model (Any):              Model used for training.
        dataset (DatasetDict):           Dataset containing train, val, and test splits.
        on_embeddings (bool):            Whether training is performed on embeddings.
        yfcc (bool):                     Whether YFCC input data was used.
        train_args (TrainingArguments):  Training arguments.
        metrics (Callable):              Function returning a dict of evaluation metrics for
                                         (predictions, labels) input.
        patience (int, optional):        Patience for early stopping. Defaults to None.
        should_profile (bool, optional): Whether PyTorch should profile. Defaults to True.

    Returns:
        AutoModel: Rrained automodel
    """
    writer = SummaryWriter()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False) #find_unused_parameters=True
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=train_args.learning_rate)
    train_data = DataLoader(dataset['train'], train_args.per_device_train_batch_size, shuffle=True,
                            pin_memory=True, num_workers=32)

    # Data loader
    model, optimizer, train_data = accelerator.prepare(loaded_model, optimizer, train_data)
    steps = len(train_data)

    # Evaluation
    prior_eval_loss = None
    current_patience = 0

    # Gradient accumulation
    grad_acc_steps = train_args.gradient_accumulation_steps if train_args.gradient_accumulation_steps \
                     is not None else 1

    # Profiler
    with generate_profiler() as prof:

        # Set to train mode
        logger.warning(f'Starting training ...')
        model.train()
        optimizer.zero_grad()

        for epoch in tqdm(range(train_args.num_train_epochs), disable=not accelerator.is_local_main_process):

            # Training loop
            combined_loss = 0
            for i, data in tqdm(enumerate(tqdm(train_data)), disable=not accelerator.is_local_main_process):
                output = model(**data)
                accelerator.backward(output.loss)
                combined_loss += output.loss

                # Gradient accumulation & Logging
                if i % grad_acc_steps == (grad_acc_steps - 1) or (i + 1) == len(train_data):
                    optimizer.step()
                    optimizer.zero_grad()

                # Logging
                if i > 0 and i % train_args.logging_steps == 0:
                    writer.add_scalar('Loss/train', output.loss, (epoch * steps) + i)
                
                # Profile GPU utilization
                if should_profile:
                    prof.step()

            # Evaluation
            eval_loss = evaluate_model(model, dataset['val'], metrics, train_args, None, yfcc, writer, epoch)

            # Save model if geolocation prediction is best
            if prior_eval_loss is None or eval_loss < prior_eval_loss:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), CURRENT_SAVE_PATH)
                prior_eval_loss = eval_loss
                current_patience = 0

            else:
                current_patience += 1
            
            # Early stopping
            if patience is not None and current_patience == patience:
                logger.warning(f'Early stopping after {patience} epochs ...')
                break
        
        # Return trained model
        return unwrapped_model