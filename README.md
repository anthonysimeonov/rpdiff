# Relational Pose Diffusion for Multi-modal Object Rearrangement
PyTorch implementation for training diffusion models to iterative de-noise the pose of an object point cloud and satisfy a geometric relationship with a scene point cloud.

---

This is the reference implementation for our paper:

### Shelving, Stacking, Hanging: Relational Pose Diffusion for Multi-modal Rearrangement
<p align="center">
<img src="./doc/all_real_results_outsc5.gif" alt="drawing" width="320">
<img src="./doc/rpdiff-just_sim.gif" alt="drawing" width="320">
</p>

[Paper](https://anthonysimeonov.github.io/) | [Video](https://youtu.be/e6_4wtuUfJw) | [Website](https://anthonysimeonov.github.io/rpdiff-multi-modal/)

[Anthony Simeonov](https://anthonysimeonov.github.io/), [Ankit Goyal*](https://imankgoyal.github.io/), [Lucas Manuelli*](http://lucasmanuelli.com/), [Lin Yen-Chen](https://yenchenlin.me/),
[Alina Sarmiento](https://www.linkedin.com/in/alina-sarmiento/), [Alberto Rodriguez](https://meche.mit.edu/people/faculty/ALBERTOR@MIT.EDU), [Pulkit Agrawal**](http://people.csail.mit.edu/pulkitag/), [Dieter Fox**](https://homes.cs.washington.edu/~fox/)

## Installation
```
git clone --recurse git@github.com:anthonysimeonov/rpdiff-dev.git
cd rpdiff

# make virtual environment

# conda below
conda create -n rpdiff-env python=3.8
conda activate rpdiff-env

# virtualenv/venv below
mkdir .envs
virtualenv -p `which python3.8` .envs/rpdiff-env
# python3.8 -m venv .envs/rpdiff-env
source .envs/rpdiff-env/bin/activate

# numpy (specific version), cython for compiled functions, and airobot for pybullet wrappers
pip install -r base_requirements.txt

# other packages + our repo (see pytorch install below for final installs)
pip install -e .

# install pytorch (see PyTorch website - we use v1.13.1 with CUDA 11.7, install command below)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

```

Some post installation steps:

Install `torch-scatter`/`torch-cluster` packages

```
# If torch version 1.13
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# If torch version 1.12 (below)
# pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
# pip install torch-cluster -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
```

Install knn_cuda utils
```
pip install https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## Environment setup
Source setup script to set environment variables (from repo root directory -- must be done in each terminal, consider adding to `.bashrc`/`.zshrc`)
```
source rpdiff_env.sh
```

For training data generation/eval/debugging, setup the meshcat visualizer in the background (i.e., use a background `tmux` terminal)
```
meshcat-server
```
Needs port `7000` to be forwarded (by default)

# Quickstart

## Download assets
Download necessary `.obj`. files, pre-trained model weights, and procedurally-generated demonstration data

```
# separately (with separate download scripts)
bash scripts/dl_model_weights.bash   # small download
bash scripts/dl_objects.bash  # small download
bash scripts/dl_train_data.bash  # large download (~80GB), slower
```

```
# all together (slower)
bash scripts/dl_rpdiff_all.bash
```

## Eval
Config files for evaluation are found in [`full_eval_cfgs` inside the `config` folder](src/rpdiff/config/full_eval_cfgs)

For `book/bookshelf`
```
cd src/rpdiff/eval
python evaluate_rpdiff.py -c book_on_bookshelf/book_on_bookshelf_withsc.yaml
```

For `mug/rack-multi`
```
cd src/rpdiff/eval
python evaluate_rpdiff.py -c mug_on_rack_multi/mug_on_rack_multi_withsc.yaml
```

For `can/cabinet`
```
cd src/rpdiff/eval
python evaluate_rpdiff.py -c can_on_cabinet/can_on_cabinet_withsc.yaml
```

There are config files for evaluating the system both with and without the additional success classifier model. 

## Training
Config files for training are found in [`train_cfgs` inside the `config` folder](src/rpdiff/config/train_cfgs)

### Pose diffusion training
For `book/bookshelf`
```
cd src/rpdiff/training
python train_full.py -c book_on_bookshelf_cfgs/book_on_bookshelf_pose_diff_with_varying_crop_fixed_noise_var.yaml
```

For `mug/rack-multi`
```
cd src/rpdiff/training
python train_full.py -c mug_on_rack_multi_cfgs/mug_on_rack_multi_pose_diff_with_varying_crop_fixed_noise_var.yaml
```

For `can/cabinet`
```
cd src/rpdiff/training
python train_full.py -c can_on_cabinet_cfgs/can_on_cabinet_pose_diff_with_varying_crop_fixed_noise_var.yaml
```

### Success classifier training

For `book/bookshelf`
```
cd src/rpdiff/training
python train_full.py -c book_on_bookshelf_cfgs/book_on_bookshelf_succ_cls.yaml
```

For `mug/rack-multi`
```
cd src/rpdiff/training
python train_full.py -c mug_on_rack_multi_cfgs/mug_on_rack_multi_succ_cls.yaml
```

For `can/cabinet`
```
cd src/rpdiff/training
python train_full.py -c can_on_cabinet_cfgs/can_on_cabinet_succ_cls.yaml
```

## Data generation
Config files for data generation are found in [`full_demo_cfgs` inside the `config` folder](src/rpdiff/config/full_demo_cfgs)

For `book/bookshelf`
```
cd src/rpdiff/data_gen
python object_scene_procgen_demos.py -c book_on_bookshelf/bookshelf_double_view.yaml
```

For `mug/rack-multi`
```
cd src/rpdiff/data_gen
python object_scene_procgen_demos.py -c mug_on_rack/mug_on_rack_multi.yaml
```

For `can/cabinet`
```
cd src/rpdiff/data_gen
python object_scene_procgen_demos.py -c can_on_cabinet/can_cabinet.yaml
```

**Post-processing**
These scripts are unfortunately *not* implemented to be easily run in parallel across many workers. Instead, we rely on the less elegant way of scaling up data generation by run the script in separate terminal windows with different seeds (which can be specified with the `-s $SEED` flag). This creates multiple separate folders with the same root dataset name and different `_seed$SEED` suffixes. Before training, these separate folders must be merged together. We also create training splits (even though the objects are already split in the data generation script) and provide utilities to combine the separate `.npz` files that are saved together into larger ``chunked'' files that are a little easier on shared NFS filesystems (for file I/O during training).

These post-processing steps are all packaged in the [`post_process_demos.py`](src/rpdiff/data_gen/post_process_demos.py) script, which takes in a `--dataset_dir` flag which should use the same name as the `experiment_name` parameter in the config file used for data gen. For example, if we generate demos with the name `bookshelf_demos`, the resulting folders in the directory where the data is saved (`data/task_demos` by default) will look like:
```
bookshelf_demos_seed0
bookshelf_demos_seed1
...
```
To combine these into a single `bookshelf_demos` folder, we then run
```
python post_process_demos.py --dataset_dir bookshelf_demos
```

# Notes on repository structure
### Config files
Config files for training, eval, and data generation are all found in the [`config`](src/rpdiff/config/) folder. These consist of `.yaml` files that inherit from a base `base.yaml` file. Utility functions for config files can be found in the [`config_util.py`](src/rpdiff/utils/config_util.py) file. Configuration parameters are loaded as nested dictionaries, which can be accessed using either standard `value = dict['key']` syntax or `value = dict.key` syntax.

### Model inference
The full rearrangement prediction pipeline is implemented in the [`multistep_pose_regression.py`](src/rpdiff/utils/relational_policy/multistep_pose_regression.py) file. This uses the trained models and the observed point clouds to iteratively update the pose of the object point cloud, while tracking the overall composed transform thus far until returning the final full transformation to execute. 

### Information flow between data generation, training, and evaluation via config files
When generating demonstration data, we give the set of demonstrations a name and post-process the demos into chunked files. During training, we must provide the path to the demos to load in the config file, and specify if these are the chunked demos or the original un-chunked demos. While training, model weights will be saved in the `model_weights/rpdiff` folder. When we evaluate the system, we similarly must specify the path to the model weights in the corresponding config file loaded when running the eval script. 

# Citing
If you find our paper or this code useful in your work, please cite our paper:
```
@article{simeonov2023rpdiff,
    author = {Simeonov, Anthony
                and Goyal, Ankit
                and Manuelli, Lucas
                and Yen-Chen, Lin
                and Sarmiento, Alina,
                and Rodriguez, Alberto
                and Agrawal, Pulkit
                and Fox, Dieter},
    title = {Shelving, Stacking, Hanging: Relational
                Pose Diffusion for Multi-modal Rearrangement},
    journal={arXiv preprint arXiv:0000.00000},
    year={2023}
}
```

# Acknowledgements
Parts of this code were built upon implementations found in the [Relational NDF repo](https://github.com/anthonysimeonov/relational_ndf), the [Neural Shape Mating repo](https://github.com/pairlab/NSM), and the [Convolutional Ocupancy Networks repo](https://github.com/autonomousvision/convolutional_occupancy_networks). Check out their projects as well!