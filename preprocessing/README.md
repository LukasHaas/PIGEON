# Dataset Preprocessing

This folder contains multiple utility functions and scripts that are either important as utility functions or run separately as scripts.

## ```dataset_preprocessing.py```

Functions to compute the labels and extract image features as part of the dataset preprocessing pipeline.

## ```embed.py```

This script contains functions used to compute image embeddings using the the domain-pretrained PIGEON and PIGEOTTO vision encoders as embedders. Embedding is performed on multiple GPUs in parallel.

## ```geo_augmentor.py```

Before training any models, this script is run on a dataset to generate all data required for label computation and multi-task training.

All data is inferred from the coordinates of the image locations.

## ```geo_utils.py```

Contains implementations to perform coordinate transformations and haversine distance computation in both numpy and PyTorch. An additional implementation was written to compute the distances of all possible combinations of two sets up points in parallel to speed up training and inference times.

## ```utils.py```

Additional utility functions.