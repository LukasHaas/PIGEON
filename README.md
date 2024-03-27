# PIGEON: Predicting Image Geolocations
This repository contains the code for the paper *PIGEON: Predicting Image Geolocations*. The paper was authored by Lukas Haas, Michal Skreta, Silas Alberti, and Chelsea Finn at Stanford University.

This repository is purely meant for the academic validation of the paper's code. Geocell shapes and coordinates, training and validation datasets, and model weights are not provided as part of this release. Please read the section *Ethical considerations* in our paper to learn more.

## Citing This Work

Please cite our work as follows:

```
@misc{haas2023pigeon,
      title={PIGEON: Predicting Image Geolocations}, 
      author={Lukas Haas and Michal Skreta and Silas Alberti and Chelsea Finn},
      year={2023},
      eprint={2307.05845},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Contents

This root folder contains the following code files. Each directory contains another README.

### ```config.py```

Definition of many different constants and paths needed in the rest of the repository.

### ```env.yml```

Conda environment used in this project.

### ```get_auxiliary_data.sh```

Script to download auxiliary data used in this project.

### ```run.py```

This file is the entry point to this project, loads and preprocesses the data, and depending on the command pretrains, finetunes, embeds, or evaluate data.
