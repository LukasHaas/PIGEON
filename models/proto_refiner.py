import json
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from typing import Dict, List, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import DatasetDict, Dataset, enable_progress_bar, disable_progress_bar, concatenate_datasets
from config import PROTO_PATH, DATASET_PATH
from models.layers import HedgeLayer
from preprocessing import haversine

# Cluster refinement model

class ProtoRefiner(nn.Module):
    """Proto-Net refinement model
    """
    def __init__(self, topk: int=5, hedge: bool=False, max_refinement: int=1000,
                 temperature: float=1.6, proto_path: str=PROTO_PATH,
                 dataset_path: str=DATASET_PATH, protos: List[Dataset]=None,
                 verbose: bool=False):
        """Proto-Net refinement model

        Args:
            topk (int, optional): number geocell candidates to consider.
                Defaults to 5.
            hedge (bool, optional): whether guesses should be hedged.
                Defaults to False.
            max_refinement (int, optional): max refinement distance in km.
                Defaults to 1000.
            temperature (float, optional): temperature influencing the softmax strength of
                the refiner probabilities. Defaults to 1.6.
            proto_path (str, optional): path to proto-cluster refinement file.
                Defaults to PROTO_PATH.
            dataset_path (str, optional): path to file containing embeddings for training dataset.
                The embeddings must be the ones produced by the SuperGuessr model for which
                guesses are refined. Defaults to DATASET_PATH.
            protos (List[Dataset], optional): directly supplied protos if needed. Defaults to None.
            verbose (bool, optional): Whether to print out processes in detail.
                Defaults to False.
        """
        super(ProtoRefiner, self).__init__()

        # Variables
        self.topk = topk
        self.hedge = hedge
        self.max_refinement = max_refinement
        self.verbose = verbose

        # Load dataset with embeddings and prototypes
        if type(dataset_path) == list:
            if len(dataset_path) > 2:
                raise NotImplementedError('Can\'t concatentate more than 2 datasets.')

            # Dataset
            data_1 = DatasetDict.load_from_disk(dataset_path[0])
            data_2 = DatasetDict.load_from_disk(dataset_path[1])
            data_1 = data_1.remove_columns(['labels_climate'])
            data_2 = data_2.remove_columns(['labels_climate'])
            self.dataset = DatasetDict(
                train=concatenate_datasets([data_1['train'], data_2['train']])
            )

        else:
            self.dataset = DatasetDict.load_from_disk(dataset_path)

        # Load prototypes
        self.proto_df = pd.read_csv(proto_path)
        self.proto_df['indices'] = self.proto_df['indices'].apply(self._load_indices)

        # Load prototype index dataframe
        self.proto_df['geocell_idx'] = self.proto_df['geocell_idx'].astype(int)
        self.num_geocells = self.proto_df['geocell_idx'].max() + 1
        self.proto_df = self.proto_df.set_index('geocell_idx')

        # Generate prototypes for every geocells
        if protos is None:
            self.load_prototypes()
        else:
            self.protos = protos

        # Hedge layer for competitive games
        if self.hedge:
            self.hedge_layer = HedgeLayer(temperature=5) # 1.4

        # Learnable parameters
        self.temperature = Parameter(torch.tensor(temperature), requires_grad=False)
        self.geo_scaling = Parameter(torch.tensor(20.), requires_grad=False)

    def _load_indices(self, index_json: str, add_index: int=None):
        """Loads protonet cluster assignments.

        Args:
            index_json (str): string of indices of samples belonging to cluster.
        """
        try:
            result = json.loads(index_json)
            if add_index is not None:
                result = [x + add_index for x in result]

            return result
             
        except TypeError:
            if self.verbose:
                print(f'Couldn\'t load a geocell.')

            return []

    def __str__(self):
        rep = 'ProtoRefiner(\n'
        rep += f'\ttopk\t\t= {self.topk}\n'
        rep += f'\thedge\t\t= {self.hedge}\n'
        rep += f'\tmax_refinement\t= {self.max_refinement}\n'
        rep += f'\ttemperature\t= {self.temperature.data.item()}\n'
        rep += f'\tgeo_scaling\t= {self.geo_scaling.data.item()}\n'
        rep += ')'
        return rep

    def forward(self, embedding: Tensor=None, geo_tensor: Tensor=None, 
                initial_preds: Tensor=None, candidate_cells: Tensor=None,
                candidate_probs: Tensor=None, cluster: Tensor=None):
        """Forward function for proto refinement model.

        Args:
            embedding (Tensor): CLIP embeddings of images.
            geo_tensor (Tensor): tensor containing multi-task regression predictions.
            initial_preds (Tensor): initial predictions.
            candidate_cells (Tensor): tensor of candidate geocell predictions.
            candidate_probs (Tensor): tensor of probabilities assigned to
                each candidate geocell. Defaults to None.
            cluster (Tensor): cluster label for training
        """
        assert self.topk <= candidate_cells.size(1), \
            '"topk" parameter must be smaller or equal to the number of geocell candidates \
             passed into the forward function.'

        if embedding.dim() == 3:
            embedding = embedding.mean(dim=1)

        # If no probabilities are passed, only consider first cell candidate
        if candidate_probs is None:
            candidate_probs = torch.zeros_like(candidate_cells)
            candidate_probs[:, 0] = 1

        # Setup variables
        guess_index = []
        preds_LLH = []
        preds_geocell = []
        loss = 0 if self.training else None
        
        # Loop over every data sample
        for i, (emb, candidates, c_probs) in enumerate(zip(
                embedding, candidate_cells, candidate_probs)):
            top_preds = []
            top_distances = []

            # Loop over every candidate cell
            for cell in candidates[:self.topk]:

                # Embedding distance
                cell_id = cell.item()
                if cell_id in [121, 650, 1859]:  # TODO: fix
                    cell_id = 1436

                cell_emb = self.protos[cell.item()]
                if cell_emb is None:
                    if self.verbose:
                        print(f'Proto dataset for geocell {cell.item()} is None.')

                    top_distances.append(torch.tensor(-100000, device='cuda'))
                    top_preds.append([0., 0.])
                    continue
                
                cell_emb = cell_emb['embedding'].to('cuda')
                logits = -self._euclidian_distance(cell_emb, emb)

                # Get top cluster and corresponding coordinates
                top_distances.append(torch.max(logits).item())
                pred_id = torch.argmax(logits, dim=-1)
                entry = self.protos[cell.item()][pred_id.item()]
                lng, lat = self._within_cluster_refinement(emb, entry)
                top_preds.append([lng, lat])

            # Temperature softmax over cluster candidates
            top_distances = torch.tensor(top_distances, device='cuda')
            probs = self._temperature_softmax(top_distances)

            # Multiply proto probabilities with geocell probabilities
            initial_guess = torch.argmax(c_probs[:self.topk]).item()
            final_probs = c_probs[:self.topk] * probs
            refined_guess = torch.argmax(final_probs).item()
            if refined_guess != initial_guess and self.verbose:
                print('\t\tRefinement changed geocell.')

            # Don't refine if refinement is more than max_refinement km
            refined_LLH = torch.tensor(top_preds[refined_guess], device='cuda')
            refined_LLH = refined_LLH.unsqueeze(0)
            initial_LLH = initial_preds[i].unsqueeze(0)
            distance = haversine(initial_LLH, refined_LLH)[0]
            if distance > self.max_refinement:
                final_probs = c_probs[:self.topk]
                if self.verbose:
                    print('\t\tCancelled refinement: distance too far.')

            # Hedge guesses in competitive games
            if self.hedge:
                preds = torch.from_numpy(np.array(top_preds)).to('cuda')
                final_probs = self.hedge_layer(preds, final_probs)
                new_guess = torch.argmax(final_probs).item()
                if refined_guess != new_guess:
                    if self.verbose:
                        print('\t\tHedging changed location prediction.')
                        print(f'\t\t{top_preds[refined_guess]} -> {top_preds[new_guess]}')
                        if new_guess == initial_guess:
                            print('\t\tHedging changed back to original geocell.')

            final_pred_id = torch.argmax(final_probs).item()
            guess_index.append(final_pred_id)
            preds_LLH.append(top_preds[final_pred_id])
            preds_geocell.append(candidates[final_pred_id])

        # Look at percent of changed predictions
        guess_index = torch.tensor(guess_index, device='cuda')
        perc_changed = (guess_index != 0).sum() / guess_index.size(0)
        print(f'Changed geocell predictions of {perc_changed * 100:.1f} % of guesses.')

        preds_LLH = torch.tensor(preds_LLH, device='cuda')
        preds_geocell = torch.tensor(preds_geocell, device='cuda')
        return loss, preds_LLH, preds_geocell
    
    def _within_cluster_refinement(self, emb: Tensor, 
                                   cluster: Dict[str, Tensor]) -> Tuple[float, float]:
        """Refines the guess even further by picking the image in a cluster that matches the best.

        Args:
            emb (Tensor): embedding of query image.
            cluster (Dict[str, Tensor]): Huggingface dataset entry.

        Returns:
            Tuple[float, float]: (lng, lat)
        """
        if cluster['count'] == 1:
            return cluster['lng'].item(), cluster['lat'].item()

        points = self.dataset['train'][cluster['indices']]
        embeddings = points['embedding'].to('cuda')
        if embeddings.size(1) == 4:
            embeddings = embeddings.mean(dim=1)

        distances = self._euclidian_distance(embeddings, emb)
        max_index = torch.argmax(distances).item()
        max_point = points['labels'][max_index]
        return max_point[0].item(), max_point[1].item()

    def load_prototypes(self) -> List[Dataset]:
        """Load prototypes used for matching.

        Returns:
            List[Dataset]: list of datasets for every geocell
        """
        print('Initializing ProtoRefiner. This might take a while ...')

        # Create progress bar
        progress_bar = tqdm(total=self.num_geocells, desc='Processing', position=0, leave=True)
        disable_progress_bar() # dataset.map progress bar, not tqdm
        
        # Multi-processing for CPU-bound (not I/O bound) prototype generation
        with ProcessPoolExecutor(max_workers=64) as executor:
            future_to_index = {executor.submit(self._get_prototypes, i): i for i in range(self.num_geocells)}
            self.protos = [None] * self.num_geocells
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    self.protos[index] = future.result()
                except Exception as exc:
                    print(f'Task {index} generated an exception: {exc}')
                
                progress_bar.update(1)
            
        # Close progress bar after completion
        progress_bar.close()
        enable_progress_bar()

        print('Initialization of ProtoRefiner complete.')

    def _get_prototypes(self, cell: int) -> Dataset:
        """Gets embedding and geo-tensor prototypes for cell.

        Args:
            cell (int): geocell index

        Returns:
            Dataset: dataset including prototypes
        """
        try:
            cell_df = self.proto_df.loc[cell]

        # Some geocells might overlap with others, causing no data points to be in the cell
        except KeyError:
            return None

        if type(cell_df) == pd.core.series.Series:
            cell_df = pd.DataFrame([cell_df])

        if len(cell_df.iloc[0]['indices']) == 0:
            return None

        data = Dataset.from_pandas(cell_df)
        data = data.map(self._compute_protos_for_cell)
        data.set_format('torch')
        return data

    def _cosine_similarity(self, matrix: Tensor, vector: Tensor) -> Tensor:
        """Computes the cosine similarity between all vectors in matrix and vector.

        Args:
            matrix (Tensor): matrix of shape (N, dim_vector)
            vector (Tensor): vector of shape (dim_vector)

        Returns:
            Tensor: cosine similarities
        """
        dot_product = torch.mm(matrix, vector.unsqueeze(1))
        matrix_norm = torch.norm(matrix, dim=1).unsqueeze(1)
        vector_norm = torch.norm(vector)
        cosine_similarities = dot_product / (matrix_norm * vector_norm)

        return cosine_similarities.flatten()

    def _euclidian_distance(self, matrix: Tensor, vector: Tensor) -> Tensor:
        """Computes the euclidian distance between all vectors in matrix and vector.

        Args:
            matrix (Tensor): matrix of shape (N, dim_vector)
            vector (Tensor): vector of shape (dim_vector)

        Returns:
            Tensor: euclidian distances
        """
        v = vector.unsqueeze(0)
        distances = torch.cdist(matrix, v)
        return distances.flatten()

    def _temperature_softmax(self, input: Tensor) -> Tensor:
        """Performs softmax with learnable temperature.

        Args:
            input (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        ex = torch.exp(input / self.temperature)
        sum = torch.sum(ex, axis=0)
        return ex / sum

    def _compute_protos_for_cell(self, example: Dict, geo: bool=False) -> Dict:
        """Computes embedding and geo prototypes.

        Args:
            example (Dict): data sample
            geo (bool, optional): whether geo tensor should be included.
                Defaults to False.

        Returns:
            Dict: modified data sample
        """
        indices = example['indices']
        entries = self.dataset['train'][indices]
        embeddings = entries['embedding']
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)

        proto_emb = embeddings.mean(dim=0)
        if geo == False:
            return {'embedding': proto_emb}
        
        proto_geo = entries['labels_multi_task'].mean(dim=0)
        return {
            'embedding': proto_emb,
            'geo_tensor': proto_geo
        }

