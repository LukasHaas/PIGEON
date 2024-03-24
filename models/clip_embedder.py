import torch
from torch import nn, Tensor
from typing import Dict, Callable
from .utils import load_state_dict
from config import PRETRAINED_CLIP, CLIP_MODEL
from transformers import CLIPProcessor, CLIPVisionModel
from PIL import Image
import torch.distributed

class CLIPEmbedding(torch.nn.Module):
    def __init__(self, model_name: str, device: str='cuda', load_checkpoint: bool=False,
                 panorama: bool=False):
        """CLIP embedding model (not trainable)

        Args:
            model_name (str): CLIP model version
            device (str, optional): where to run the model. Defaults to 'cuda'.
            load_checkpoint (bool, optional): whether to load checkpoint from
                CLIP_SERVING path. Defaults to True.
            panorama (bool): if four images should be embedded.
                Defaults to False.
        """
        super().__init__()
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.clip_model = CLIPVisionModel.from_pretrained(CLIP_MODEL)
        self.panorama = panorama

        # Load checkpoint if required
        if load_checkpoint:
            state_dict = torch.load(model_name, map_location=torch.device('cuda')) # '../' + PRETRAINED_CLIP
            load_state_dict(self.clip_model.base_model, state_dict, embedder=True)
            print('Loaded embedder from checkpoint:', model_name)

        if type(device) == str:
            self.clip_model = self.clip_model.to(self.device)
        else:
            self.clip_model = self.clip_model.cuda(self.device)

        self.eval()

    def _get_embedding(self, image: Image) -> Tensor:
        """Computes embedding for a single image.

        Args:
            image (Image): jpg image

        Returns:
            Tensor: embedding
        """
        with torch.no_grad():
            if isinstance(image, Tensor) == False:
                inputs = self.processor(images=image, return_tensors='pt')
                pixel_values = inputs['pixel_values']
            else:
                pixel_values = image

            if type(self.device) == str:
                pixel_values = pixel_values.to(self.device)
            else:
                pixel_values = pixel_values.cuda(self.device)
        
            outputs = self.clip_model.base_model(pixel_values=pixel_values)
            cls_token_embedding = outputs.last_hidden_state
            cls_token_embedding = torch.mean(cls_token_embedding, dim=1)
            return cls_token_embedding

    def _pre_embed_hook(self) -> Callable:
        """Hook to store forward activations of a specific layer.

        Returns:
            Callable: The hook to be registered on a module's forward function.
        """
        def hook(model, input, output):
            self.pre_embed_outputs = output[0]

        return hook

    def forward(self, image: Dict) -> Tensor:
        """Computes forward pass to generate embeddings.

        Args:
            images (Dict): _description_

        Returns:
            Tensor: _description_
        """
        #if isinstance(image, Tensor):
        return self._get_embedding(image)
    
        # UNCOMMENT FOR PIGEON

        # if 'image_2' not in kwargs:
        #     return self._get_embedding(image)

        # cols = ['image', 'image_2', 'image_3', 'image_4']

        # embeddings = []
        # for col in cols:
        #     embedding = self._get_embedding(kwargs[col])
        #     embeddings.append(embedding)

        # return torch.stack(embeddings, dim=1)