# Finetuning Datasets

This folder contains the following code files.

## ```embed_dataset.py```

Used when creating many embeddings in parallel. Allows for the preprocessing of images on the CPU while the GPU embeds prior image batches.

## ```finetune_dataset.py```

Contains functions used in generating the dataset on which PIGEON is trained (prediction head training).

## ```yfcc_dataset.py```

Dataset used to train PIGEOTTO (prediction head training).

