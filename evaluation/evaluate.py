import torch
from dataset_creation.benchmark import BenchmarkDataset
from transformers import AutoModelForImageClassification, CLIPVisionModel
from config import *
from models import SuperGuessr, ProtoRefiner, load_state_dict
from torchsummary import summary
from .metrics import compute_geoguessr_metrics
from training import evaluate_model

def evaluate(model: str, dataset: BenchmarkDataset, yfcc: bool, landmarks: bool,
             base_model: str=None, heading: bool=False,
             refine: bool=True) -> AutoModelForImageClassification:
    """Evaluates a model on a dataset split.

    Note:
        Assumes base model is CLIP model specified in config "CLIP_MODEL" constant.

    Args:
        model (str): model or prediction head path to be loaded from disk.
        dataset (DatasetDict): dataset.
        yfcc (bool): whether the model was trained on yfcc data.
        landmarks (bool): whether landmarks should be used in prototype generation.
        base_model (str): base model path if a pretrained based model was used.
        multi_task (bool): whether the model should be trained in a multi-task setting.
            Defaults to True.
        heading (bool): whether the model uses the compass heading or not.
            Defaults to False.
        refine (bool, optional): whether a guess refinement model should be used.
            Defaults to True.

    Returns:
        AutoModel: finetuned model.
    """

    # Load model
    embed_model = CLIPVisionModel.from_pretrained(CLIP_MODEL) if base_model is not None else None
    if base_model is not None and base_model != CLIP_MODEL:
        state_dict = torch.load(base_model, map_location=torch.device('cuda'))
        load_state_dict(embed_model, state_dict)
        print(f'Initialized base model with weights from: {base_model}')

    full_model = SuperGuessr(embed_model, panorama=True, hierarchical=False,
                             multi_task=False, heading=heading, freeze_base=True,
                             yfcc=yfcc, num_candidates=50)
    full_model.load_state('saved_models/WorldCLIP_head.model')
    full_model.load_state(model)
    full_model.to('cuda')
    summary(full_model)
    print(full_model)

    # Guess refinement
    refiner = None
    if refine:
        
        # Constants
        proto_model_path = PROTO_MODEL_YFCC_PATH if yfcc else PROTO_MODEL_PATH
        proto_path = PROTO_PATH_YFCC if yfcc else PROTO_PATH
        dataset_path = DATASET_PATH_YFCC if yfcc else DATASET_PATH

        if landmarks:
            proto_path = PROTO_PATH_LANDMARKS
            proto_model_path = PROTO_MODEL_LANDMARKS_PATH
            dataset_path = [DATASET_PATH_YFCC, DATASET_PATH_LANDMARKS]

        protos = None
        try:
            ref = torch.load(proto_model_path, map_location='cuda')
            protos = ref.protos
        except FileNotFoundError:
            pass

        if protos is None:
            refiner = ProtoRefiner(20, False, 10000, proto_path=proto_path, dataset_path=dataset_path,
                                   temperature=1)
            torch.save(refiner, proto_model_path)
        else:
            # StreetView: 5, 1000km max restriction, temp=1.6
            # YFCC + landmarks: 40, no max restriction, temp=0.6
            refiner = ProtoRefiner(40, False, 100000, proto_path=proto_path, dataset_path=dataset_path,
                                   protos=protos, temperature=0.6, verbose=False)

        print(refiner)

    # Perform evaluation
    _ = evaluate_model(full_model, dataset, compute_geoguessr_metrics, TRAIN_ARGS, refiner)
    return model