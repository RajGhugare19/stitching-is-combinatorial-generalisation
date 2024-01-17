# Closing the Gap between TD Learning and Supervised Learning -- A Generalisation Point of View
[Raj Ghugare](https://rajghugare19.github.io/), $\quad$ [Matthieu Geist](https://homangab.github.io/), $\quad$ [Glen Berseth](https://neo-x.github.io/)<sup>\*</sup>, $\quad$ [Benjamin Eysenbach](https://ben-eysenbach.github.io/)<sup>\*</sup>

<sup>\*</sup> Equal advising.

## Installation

Create virtual environment named `env_alm` using command:<br>
```sh
python3 -m venv env_dt
```

Install all the packages used to run the code using the `requirements.txt` file: <br>
```sh
pip install -r requirements.txt
```

## Training

To train an RvS (decision-mlp) agent on pointmaze-umaze using temporal data augmentation, with $\epsilon=0.5$ and $K=40$:<br> 
```sh
python train_dmlp.py dataset_name=pointmaze-umaze-v0 augment_data=True nclusters=40
```

To train an DT (decision-transformer) agent on pointmaze-umaze using temporal data augmentation, with $\epsilon=0.5$ and $K=40$:<br> 
```sh
python train_dt.py dataset_name=pointmaze-umaze-v0 augment_data=True nclusters=40
```

## Datasets

To download the pretrained datasets, visit [this google drive link](https://drive.google.com/drive/folders/1j8Ok2UMYSqfIQReuE6csf1nMoI1s25K-?usp=sharing).

To collect the pointmaze-large dataset with $1e^6$ transitions and seed 1:<br> 
```sh
python collect_pointmaze_data.py pointmaze-large-v0 1 1000000
```

To collect the antmaze-large dataset with $1e^6$ transitions and seed 1:<br> 
```sh
python collect_antmaze_data.py antmaze-umaze-v0 1 1000000
```

## Acknowledgment
Our codebase has been build using/on top of the following codes. We thank the respective authors for their awesome contributions.
- [NanoGPT](https://github.com/karpathy/nanoGPT)<br>
- [min-decision-transformer](https://github.com/nikhilbarhate99/min-decision-transformer)<br>

## Correspondence

If you have any questions or suggestions, please reach out to me via raj.ghugare@mila.quebec.
