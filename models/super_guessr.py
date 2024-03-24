import torch
import pandas as pd
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from collections import namedtuple
from preprocessing import haversine_matrix, smooth_labels
from models.layers import PositionalEncoder
from models.utils import ModelOutput
from config import *

#  Named Tuples
MultiTaskPredictions = namedtuple('MultiTaskPredictions', 'loss_reg preds_mt loss_climate \
                                   preds_climate loss_month preds_month')

# Constants
NUM_MULTI_TASK_VARIABLES = 6
REGRESSION_LOSS_SCALING = 8

NUM_CLIMATES = 28
CLIMATE_LOSS_SCALING = 2

NUM_MONTHS = 12
MONTHS_LOSS_SCALING = 1

NUM_ATTENTION_HEADS = 16

GEOGUESSR_HEADING_SINGLE = [0.0, 1.0]
GEOGUESSR_HEADING_MULTI = [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]

class SuperGuessr(nn.Module):
    def __init__(self, base_model: nn.Module, panorama: bool=False, hierarchical: bool=False, 
                 should_smooth_labels: bool=False, multi_task: bool=False, heading: bool=False,
                 yfcc: bool=False, serving: bool=False, freeze_base: bool=False,
                 num_candidates: int=5, embed_dim: int=CLIP_EMBED_DIM, **kwargs):
        """Initializes a location prediction model classification model.

        Args:
            base_model (nn.Module): vision encoder model on top of which the location
                predictor is built. If None, assumes that model is run directly on embeddings.
            panorama (bool, optional): whether four images are passed in as a panorama.
                Defaults to False.
            hierarchical (bool, optional): whether to use a hierarchical model to combine embeddings.
                Defaults to False.
            should_smooth_labels (bool, optional): If labels should be smoothed. Label smoothing
                works only in classification mode and penalizes guesses based on the actual distance
                of the geocell to the correct location instead of penalizing equally across all
                incorrect cells. Label smoothing also makes the prediction task easier. 
                Defaults to False.
            multi_task (bool, optional): Whether the model should create prediction heads to
                predict other geographic variables including population density, temperature, and
                precipitation. Defaults to False.
            heading (bool, optional): Whether to incorporate compass heading during training.
                Defaults to False.
            yfcc (bool, optional): Whether the model is being trained on YFCC data. Defaults to False.
            serving (bool, optional): Whether model is instantiated for serving purposes only. If set to 
                True, outputs solely lng/lat predictions in eval mode.
            freeze_base (bool, optional): If the weights of the base model should be frozen.
                Defaults to False.
            num_candidates (int, optional): Number of geocell candidates to
                produce for refinement. Defaults to 5.
            embed_dim (int, optional): Embedding dimension if base model is None. Defaults to 1024.
        """
        super(SuperGuessr, self).__init__()

        # List kwargs
        if len(kwargs) > 0:
            print(f'Not using keyword arguments: {list(kwargs.keys())}')

        # Save variables
        self.base_model = base_model
        self.panorama = panorama
        self.hidden_size = embed_dim
        self.serving = serving
        self.should_smooth_labels = should_smooth_labels
        self.multi_task = multi_task
        self.heading = heading
        self.yfcc = yfcc
        self.freeze_base = freeze_base
        self.hierarchical = hierarchical
        self.num_candidates = num_candidates

        # Setup
        self._set_hidden_size()
        geocell_path = GEOCELL_PATH_YFCC if self.yfcc else GEOCELL_PATH
        self.lla_geocells = self.load_geocells(geocell_path)
        self.num_cells = self.lla_geocells.size(0) 

        # Input dimension for cell layer
        self.input_dim = self.hidden_size
        if self.heading and not (self.panorama and not self.hierarchical):
            print('Model includes heading as feature.')
            self.input_dim += 2

        # Self-attention layer
        if self.hierarchical:
            print('Number of attention heads:', NUM_ATTENTION_HEADS)
            self.heading_pad = NUM_ATTENTION_HEADS - 2 if self.heading else 0
            self.pos_encoder = PositionalEncoder(self.input_dim + self.heading_pad)
            self.self_attn = nn.MultiheadAttention(self.input_dim + self.heading_pad,
                                                   NUM_ATTENTION_HEADS,
                                                   dropout=0.1,
                                                   batch_first=True)
            self.relu = nn.ReLU()
            
        # Cell layer
        self.cell_layer = nn.Linear(self.input_dim, self.num_cells)
        self.softmax = nn.Softmax(dim=-1)

        # Multi-task
        if self.multi_task:
            print('Model is multi-task.')

            # Regression tasks
            self.multi_task_head = nn.Linear(self.hidden_size, NUM_MULTI_TASK_VARIABLES) # self.input_dim
            self.loss_fnc_mt = nn.MSELoss(reduction='mean')

            # Climate zone
            self.climate_layer = nn.Linear(self.input_dim, NUM_CLIMATES)
            self.loss_fnc_climate = nn.CrossEntropyLoss()

            # Month prediction
            if not self.yfcc:
                self.month_layer = nn.Linear(self.input_dim, NUM_MONTHS)
                self.loss_fnc_month = nn.CrossEntropyLoss()

        # Freeze / load parameters
        self._freeze_params()

        # Loss
        self.loss_fnc = nn.CrossEntropyLoss()
        print(f'Initialized SuperGuessr classification model with {self.num_cells} geocells.')        

    def _set_hidden_size(self):
        """
        Determines the hidden size of the model
        """
        if self.base_model is not None:
            try:
                self.hidden_size = self.base_model.config.hidden_size
                self.mode = 'transformer'

            except AttributeError:
                self.hidden_size = self.base_model.config.hidden_sizes[-1]
                self.mode = 'convnext'

    def _freeze_params(self):
        """Freezes model parameters depending on mode
        """
        if self.base_model is not None:
            if self.freeze_base:
                for param in self.base_model.parameters():
                    param.requires_grad = False

            # Load parameters and freeze relevant parameters
            elif 'clip-vit' in self.base_model.config._name_or_path and not self.serving:
                head = CLIP_PRETRAINED_HEAD_YFCC if self.yfcc else CLIP_PRETRAINED_HEAD
                self.load_state(head)
                print(f'Initialized model parameters from model: {head}')
                for param in self.base_model.vision_model.encoder.layers[:-1].parameters():
                    param.requires_grad = False

    def load_geocells(self, path: str) -> Tensor:
        """Loads geocell centroids and converts them to ECEF format

        Args:
            path (str, optional): path to geocells. Defaults to GEOCELL_PATH.

        Returns:
            Tensor: ECEF geocell centroids
        """
        geo_df = pd.read_csv(path)
        lla_coords = torch.tensor(geo_df[['lng', 'lat']].values)
        lla_geocells = nn.parameter.Parameter(data=lla_coords, requires_grad=False)
        return lla_geocells

    def _move_to_cuda(self, pixel_values: Tensor=None, embedding: Tensor=None, 
                      heading: Tensor=None, labels: Tensor=None, labels_clf: Tensor=None,
                      labels_multi_task: Tensor=None, labels_climate: Tensor=None,
                      labels_month: Tensor=None):
        """Moves supplied tensors to device.

        Args:
            pixel_values (Tensor, optional): preprocessed images pixel values.
            embedding (Tensor, optional): image embeddings if no pass through
                a base model is performed.
            heading (Tensor, optional): sin and cos of compass heading.
            labels (Tensor, optional): coordinates or classification labels.
            labels_clf (Tensor, optional): index of ground truth geocell.
            labels_multi_task (Tensor, optional): 6 labels per sample in the form
                (elevation, population, temperature avg, temperature diff,
                precipitation avg, precipitation_diff).
        """
        device = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        if not self.training and device == 'cuda':
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)

            if embedding is not None:
                embedding = embedding.to(device)

            if heading is not None:
                heading = heading.to(device)

            if labels is not None:
                labels = labels.to(device)

            if labels_clf is not None:
                labels_clf = labels_clf.to(device)

            if labels_multi_task is not None:
                labels_multi_task = labels_multi_task.to(device)

            if labels_climate is not None:
                labels_climate = labels_climate.to(device)

            if labels_month is not None:
                labels_month = labels_month.to(device)

        return pixel_values, embedding, heading, labels, labels_clf, \
                labels_multi_task, labels_climate, labels_month
               
    def load_state(self, path: str):
        """Loads weights from path and applies them to the model.

        Args:
            path (str): path to model weights
        """
        own_state = self.state_dict()
        state_dict = torch.load(path, map_location=torch.device('cuda'))
        for name, param in state_dict.items():
            if name not in own_state:
                print(f'Parameter {name} not in model\'s state.')
                continue

            if isinstance(param, Parameter):
                param = param.data

            own_state[name].copy_(param)

    def _assert_requirements(self, pixel_values: Tensor=None, embedding: Tensor=None, 
                             heading: Tensor=None):
        """Checks assertions related to input.

        Args:
            pixel_values (Tensor, optional): preprocessed images pixel values.
            embedding (Tensor, optional): image embeddings if no pass through
                a base model is performed.
            heading (Tensor, optional): sin and cos of compass heading.
        """
        if self.training and self.heading:
            assert heading is not None, 'If model is in heading mode, headings must be supplied \
                                         during training.'

        if self.base_model is not None:
            assert pixel_values is not None, 'Parameter "pixel_values" must be supplied if model \
                                              has a base model.'
        else:
            assert embedding is not None, 'Parameter "embedding" must be supplied if model \
                                           does not have a base model.'

    def _concat_heading(self, input: Tensor, heading: Tensor=None) -> Tensor:
        """Concatenates the input tensor with the heading tensor.

        Args:
            input (Tensor): image embedding
            heading (Tensor, optional): headings. If None, headings are assumed
                to be pointing north which is the case while playing Geoguessr.
                Defaults to None.

        Returns:
            Tensor: concatenated image emebdding with heading
        """
        if self.panorama and not self.hierarchical:
            return input

        num_samples = input.size(0)
        shape = (num_samples, 1)
        default_input = GEOGUESSR_HEADING_SINGLE

        # Four images
        if self.panorama:
            input = input.reshape((num_samples, 4, -1))
            default_input = GEOGUESSR_HEADING_MULTI
            shape = (num_samples, 1, 1)
            if heading is not None:
                heading = heading.reshape((num_samples, 4, -1))

        elif heading is not None:
            heading = heading[:, 0]
        
        # Create default input
        if heading is None:
            heading = torch.tensor(default_input, device='cuda')
            heading = heading.repeat(*shape)

        output = torch.cat((input, heading), dim=-1)
        return output

    def _to_one_hot(self, tensor: Tensor) -> Tensor:
        """Convert a scalar tensor to a one-hot encoded tensor.
        
        Args:
            tensor (torch.Tensor): The input tensor.
            num_classes (int): The number of classes for one-hot encoding.
            
        Returns:
            Tensor: The one-hot encoded tensor.
        """
        if tensor.dim() == 0:
            one_hot = torch.zeros(self.num_cells, device=tensor.device)
            one_hot[tensor.item()] = 1
            return one_hot
        else:
            return tensor

    def _multi_task_predictions(self, embedding: Tensor=None, labels_multi_task: Tensor=None,
                                labels_climate: Tensor=None, labels_month: Tensor=None) -> MultiTaskPredictions:
        """Computes multi-task losses and predictions.

        Args:
            embedding (Tensor, optional): Last hidden layer embedding. Defaults to None.
            labels_multi_task (Tensor, optional): Multi-task labels. Defaults to None.
            labels_climate (Tensor, optional): Climate labels. Defaults to None.
            labels_month (Tensor, optional): Month labels. Defaults to None.

        Returns:
            MultiTaskPredictions: All multi task losses and predictions.
        """
        loss_reg, preds_mt = 0, None
        loss_climate, preds_climate = 0, None
        loss_month, preds_month = 0, None

        if self.multi_task:
            preds_mt = self.multi_task_head(embedding)
            preds_climate = self.climate_layer(embedding)

            if not self.yfcc:
                preds_month = self.month_layer(embedding)

            if not self.serving:
                loss_reg = self.loss_fnc_mt(preds_mt, labels_multi_task) * REGRESSION_LOSS_SCALING
                labels_climate = labels_climate.to(dtype=torch.float32)
                loss_climate = self.loss_fnc_climate(preds_climate, labels_climate) * CLIMATE_LOSS_SCALING
                if not self.yfcc:
                    loss_month = self.loss_fnc_month(preds_month, labels_month) * MONTHS_LOSS_SCALING

        mtps = MultiTaskPredictions(loss_reg, preds_mt, loss_climate, preds_climate, loss_month, preds_month)
        return mtps

    def forward(self, pixel_values: Tensor=None, embedding: Tensor=None, 
                heading: Tensor=None, labels: Tensor=None, labels_clf: Tensor=None,
                labels_multi_task: Tensor=None, labels_climate: Tensor=None,
                labels_month: Tensor=None, index: Tensor=None) -> ModelOutput:
        """Computes forward pass through network.

        Args:
            pixel_values (Tensor, optional): preprocessed images pixel values.
            embedding (Tensor, optional): image embeddings if no pass through
                a base model is performed.
            heading (Tensor, optional): sin and cos of compass heading.
            labels (Tensor, optional): coordinates or classification labels.
            labels_clf (Tensor, optional): index of ground truth geocell.
            labels_multi_task (Tensor, optional): 6 labels per sample in the form
                (elevation, population, temperature avg, temperature diff,
                precipitation avg, precipitation_diff).
            labels_climate (Tensor, optional): one of 28 labels for Koppen-Geiger
                climate zone.
            labels_month (Tensor, optional): month the image was taken (0 index)

        Returns:
            ModelOutput: named tuple of model outputs. If serving, will be tuple.

        Note:
            If a base model was specified, pixel_values need to be supplied.
            Otherwise, if the model is working directly on embeddings,
            embedding must not be None.
        """
        self._assert_requirements(pixel_values, embedding, heading)

        # Device
        pixel_values, embedding, heading, labels, labels_clf, labels_multi_task, labels_climate, labels_month = \
            self._move_to_cuda(pixel_values, embedding, heading, labels, labels_clf,
                               labels_multi_task, labels_climate, labels_month)

        # If panorama, (N, 4 * pixels) -> (N * 4, pixels)
        if self.panorama and pixel_values is not None:
            num_samples = pixel_values.size(0)
            pixel_values = pixel_values.reshape((num_samples * 4, 3, 336, 336))
        
        # Feed through base model
        if self.base_model is not None and pixel_values is not None:
            if pixel_values.dim() > 4:
                pixel_values = pixel_values.squeeze(1)
                
            embedding = self.base_model(pixel_values=pixel_values)
            if self.mode == 'transformer':
                embedding = embedding.last_hidden_state
                embedding = torch.mean(embedding, dim=1)

            else:
                embedding = embedding.pooler_output

            # (N * 4, pixels) -> (N, 4, pixels)
            if self.panorama:
                embedding = embedding.reshape((num_samples, 4, -1))

        layer_input = embedding

        # Concatenate heading to embeddings
        if self.heading:
            layer_input = self._concat_heading(layer_input, heading)

        # Handle four image input
        if self.panorama:

            # Hierarchical architecture
            if self.hierarchical:

                # Positional encoding
                if self.heading:
                    zeros = torch.zeros((layer_input.size(0), 4, self.heading_pad), device='cuda')
                    layer_input = torch.cat((layer_input, zeros), dim=-1)

                layer_input = self.pos_encoder(layer_input)

                # Multi-head self attention
                output = self.self_attn(layer_input,
                                        layer_input,
                                        layer_input,
                                        need_weights=False)[0]

                # Pool (CLS) and remove zero concats
                output = output[:, 0]

            # Average embeddings
            else:
                output = layer_input.mean(dim=1)

        # Single Image
        elif layer_input.size(1) == 4:
            output = embedding[:, 0]

        else:
            output = embedding

        # Linear layer
        logits = self.cell_layer(output)
        geocell_probs = self.softmax(logits)            

        # Multi-task layers
        mt = self._multi_task_predictions(output, labels_multi_task, labels_climate, labels_month)

        # Compute coordinate prediction
        geocell_preds = torch.argmax(geocell_probs, dim=-1)
        pred_LLH = torch.index_select(self.lla_geocells.data, 0, geocell_preds)
        label_probs = self._to_one_hot(labels_clf) # labels_clf if normal

        # Get top 'num_candidates' geocell candidates
        geocell_topk = torch.topk(geocell_probs, self.num_candidates, dim=-1)

        # Serving
        if not self.training and self.serving:
            if self.multi_task:
                return pred_LLH, geocell_topk, mt.preds_mt, embedding
            else:
                return pred_LLH, geocell_topk, embedding
        
        # Soft labels based on distance
        if self.should_smooth_labels:
            distances = haversine_matrix(labels, self.lla_geocells.data.t())
            label_probs = smooth_labels(distances)
            
        # Loss
        loss_clf = self.loss_fnc(logits, label_probs)
        loss = loss_clf
        if self.multi_task:
            loss = loss_clf + mt.loss_reg + mt.loss_climate + mt.loss_month

        # Results
        output = ModelOutput(loss, loss_clf, mt.loss_reg, mt.loss_climate, mt.loss_month, \
                             pred_LLH, geocell_preds, mt.preds_mt, mt.preds_climate, mt.preds_month, \
                             geocell_topk, embedding)
        return output


    def __str__(self):
        rep = 'SuperGuessr(\n'
        rep += f'\tbase_model\t= {self.base_model is not None}\n'
        rep += f'\tpanorama\t= {self.panorama}\n'
        rep += f'\thierarchical\t= {self.hierarchical}\n'
        rep += f'\tmulti-task\t= {self.multi_task}\n'
        rep += f'\tyfcc\t\t= {self.yfcc}\n'
        rep += f'\tembedding_size\t= {self.hidden_size}\n'
        rep += f'\tinput_dim\t= {self.input_dim}\n'
        rep += f'\tnum_geocells\t= {self.num_cells}\n'
        rep += f'\tlabel_smoothing\t= {self.should_smooth_labels}\n'
        rep += f'\tuses_headings\t= {self.heading}\n'
        rep += f'\tfreeze_base\t= {self.freeze_base}\n'
        rep += f'\tserving\t\t= {self.serving}\n'
        rep += ')'
        return rep
