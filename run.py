import os
import torch
import argparse
import logging
from multiprocessing import set_start_method
from training import finetune_model, finetune_on_embeddings, pretrain
from dataset_creation.pretrain import PretrainDataset, PretrainDatasetYFCC
from dataset_creation.finetune import generate_finetune_dataset
from dataset_creation.benchmark import BenchmarkDataset
from preprocessing import preprocess
from evaluation import evaluate
from datasets import DatasetDict, concatenate_datasets
from models import VITEmbedding, CLIPEmbedding
from config import *

logger = logging.getLogger('run')

def parse_list(input_list_str):
    return input_list_str.split(',')

argp = argparse.ArgumentParser()
argp.add_argument('function',
    help='Whether to pretrain, finetune or evaluate a model',
    choices=['pretrain', 'finetune', 'embed', 'evaluate']
)

argp.add_argument('name',
    help='Name of the Huggingface model to finetune or path to trained model.'
)

argp.add_argument('-l', '--load', 
    help='Comma-separated list of processed dataset path.',
    default=None,
    type=parse_list
)

argp.add_argument('-b', '--base', 
    help='Path to base model.',
    default=None
)

argp.add_argument('-s', '--sample', 
    help='How many examples to sample for training.',
    default=None
)

argp.add_argument('-a', '--auxiliary', 
    help='Whether to use auxiliary information for pretraining.',
    action='store_true',
    default=False
)

argp.add_argument('-t', '--test',
    help='Set flag to evaluate on test set.',
    action='store_true',
    default=False
)

argp.add_argument('-c', '--classification',
    help='Set flag to train in classifcation setup.',
    action='store_true',
    default=True
)

argp.add_argument('-m', '--multitask',
    help='Set flag to train in multi-task setup.',
    action='store_true',
    default=False
)

argp.add_argument('--heading',
    help='Set flag to train with compass headings.',
    action='store_true',
    default=False
)

argp.add_argument('-r', '--resume',
    help='Resume training from checkpoint.',
    action='store_true',
    default=False
)

argp.add_argument('--yfcc',
    help='Set flag to train with YFCC instead of StreetView data.',
    action='store_true',
    default=False
)

argp.add_argument('--landmarks',
    help='Set flag if landmark data was added to the training mix.',
    action='store_true',
    default=False
)

def main():

    # Setup
    args = argp.parse_args()
    mode = 'classification' if args.classification else 'regression'
    logger.warning(f'Task: {args.function.capitalize()} Pigeon(\"{args.name}\") via geospatial {mode}.')

    # Constants
    DATASET_PATH = 'data/hf_new_VIT' if args.sample is None else f'data/hf_dataset_{args.sample}'

    # Load dataset
    if args.function == 'pretrain':
        if args.yfcc:
            dataset = PretrainDatasetYFCC.generate(PRETRAIN_METADATA_PATH_YFCC, args.auxiliary)
        else:
            dataset = PretrainDataset.generate(PRETRAIN_METADATA_PATH, args.auxiliary)

    elif args.load is None or len(args.load) == 0:

        # Generate HF dataset
        metadata = METADATA_PATH_YFCC if args.yfcc else METADATA_PATH
        images = IMAGE_PATH_YFCC if args.yfcc else IMAGE_PATH
        if args.landmarks:
            metadata = METADATA_PATH_LANDMARKS
            images = IMAGE_PATH_LANDMARKS

        dataset = generate_finetune_dataset(sample=args.sample, metadata_path=metadata,
                                            image_path=images)

        # Embedding model
        embedder = None
        if args.function == 'embed':
            if 'vit' not in args.name:
                print('Using CLIP embedder.')
                embedder = CLIPEmbedding(args.name, load_checkpoint=True, panorama=(not args.yfcc))
            else:
                print('Using ViT embedder.')
                embedder = VITEmbedding(args.named)

        # Preprocess
        geocells = GEOCELL_PATH_YFCC if args.yfcc else GEOCELL_PATH
        dataset_path = DATASET_PATH_YFCC if args.yfcc else DATASET_PATH
        if args.landmarks:
            dataset_path = DATASET_PATH_LANDMARKS

        dataset = preprocess(dataset, geocells, embedder, multi_task=args.multitask)  
        dataset.save_to_disk(dataset_path)

    elif len(args.load) > 1:

        # Load datasets
        datasets = [DatasetDict.load_from_disk(x) for x in args.load]

        # Concatenate datasets
        dataset = concatenate_datasets([x['train'] for x in datasets])
        dataset = DatasetDict(
            train=dataset,
            val=datasets[0]['val'],
            test=datasets[0]['test']
        )

        print('New training dataset length:', len(dataset['train']))
    
    elif os.path.isdir(args.load[0]):
        dataset = DatasetDict.load_from_disk(args.load[0])

    else:
        dataset = BenchmarkDataset(args.load[0])

    # Load or train model 
    if args.function == 'finetune':
        if args.resume:
            raise NotImplementedError(f'Resuming from checkpoint not supported.')

        finetune_model(args.name, dataset, multi_task=args.multitask, heading=args.heading, yfcc=args.yfcc)

    elif args.function == 'embed':
        if args.resume:
            raise NotImplementedError(f'Resuming from checkpoint not supported.')

        finetune_on_embeddings(dataset, multi_task=args.multitask, heading=args.heading, yfcc=args.yfcc)

    elif args.function == 'evaluate':
        if args.yfcc == False and isinstance(dataset, BenchmarkDataset) == False:
            dataset = dataset['test'] if args.test else dataset['val']

        evaluate(args.name, dataset, yfcc=args.yfcc, base_model=args.base, refine=True,
                 landmarks=args.landmarks)

    elif args.function == 'pretrain':
        pretrain_args = PRETAIN_ARGS_YFCC if args.yfcc else PRETAIN_ARGS
        pretrain(args.name, dataset, train_args=pretrain_args, resume=args.resume)

    else:
        raise NotImplementedError(f'Mode {args.function} is not implemented.')

if __name__ == '__main__':
    set_start_method('spawn')
    main()
