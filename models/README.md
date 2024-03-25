# Models

This folder contains the following code files.

## ```clip_embedder.py```

Used to compute embeddings given a pre-trained CLIP vision encoder and one or multiple images.

## ```layers/hedge.py```

Additional PyTorch layer performing guess hedging in competitive Geoguessr games. This layer is not used in our final model because it did not increase median or average performance. It does, however, help in preventing grave guessing mistakes.

## ```layers/positional_encoder.py```

Custom positional transformer encoding used in experimental model versions combining multiple image embeddings from a panorama via a hierarchical transformer architecture. Simply averaging embeddings worked best, however.

## ```proto_refiner.py```

Hierarchical cluster refinement model. Takes a set of geocell candidates, associated probabilities, and a reference dataset (embedded) as input and recomputes the most likely geocell and exact location based on visual similarity between the query image and the reference images.

## ```super_guessr.py```

The (multi-task) prediction head, mapping an image or embedding to "unrefined" coordinate predictions via our Haversine loss.

## ```utils.py```

Utility functions specific to our models.
