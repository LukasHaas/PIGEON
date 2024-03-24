from transformers import Trainer
from datasets import Dataset
from typing import Any, Tuple, Dict
from torch.nn.parameter import Parameter
from collections import namedtuple

ModelOutput = namedtuple('ModelOutput', 'loss loss_clf loss_reg loss_climate loss_month \
                         preds_LLH preds_geocell preds_mt preds_climate preds_month \
                         top5_geocells embedding')

def predict(model: Any, dataset: Dataset) -> Tuple:
    """Makes predictions given a Huggingface model.

    Args:
        model (Any): trained model.
        dataset (Dataset): dataset.

    Returns:
        Tuple: prediction tuple.
    """
    trainer = Trainer(model=model)
    return trainer.predict(dataset)

def load_state_dict(self, state_dict: Dict, embedder: bool=False):
    """Loads parameters in state_dict into model wherever possible

    Args:
        state_dict (Dict): model parameter dict
        embedder (bool, optional): whether loading state for the CLIP emebdder.
            Defaults to False.
    """
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if embedder and 'base_model' in name:
            name = '.'.join(name.split('.')[1:])

        if name not in own_state:
            print(f'Parameter {name} not in model\'s state.')
            continue

        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        own_state[name].copy_(param)