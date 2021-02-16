# Continual Learning for Graphs

## Description
This is a Python library to easily experiment with [Deep Graph Networks](https://arxiv.org/abs/1912.12693) (DGNs) in a continual learning setting. The repository is adapted from [PyDGN](https://github.com/diningphil/PyDGN)

## Installation:
(We assume **git** and **Miniconda/Anaconda** are installed)

First, make sure gcc 5.2.0 is installed: ``conda install -c anaconda libgcc=5.2.0``. Then, ``echo $LD_LIBRARY_PATH`` should always contain ``:/home/[your user name]/miniconda3/lib``. Then run from your terminal the following command:

    source install.sh [<your_cuda_version>]

Where `<your_cuda_version>` is an optional argument that can be either `cpu`, `cu92`, `cu101`, `cu102` or `cu110` for Pytorch 1.7.0. If you do not provide a cuda version, the script will default to `cpu`. The script will create a virtual environment named `cl_dgn`, with all the required packages needed to run our code. **Important:** do NOT run this command using `bash` instead of `source`!

Remember that [PyTorch MacOS Binaries dont support CUDA, install from source if CUDA is needed](https://pytorch.org/get-started/locally/)

## Usage:

### Always preprocess your datasets before launching the experiments
For MNIST,CIFAR10, and OGBG-PPA, we will automatically use the same data splits provided in the literature (see Pytorch Geometric)

#### MNIST
    python build_dataset.py --config-file CONFIGS/config_DATA_MNIST.yml

#### CIFAR10
    python build_dataset.py --config-file CONFIGS/config_DATA_CIFAR10.yml

#### OGBG-PPA
    python build_dataset.py --config-file CONFIGS/config_DATA_OGBG-PPA.yml

### Launch an experiment in debug mode (example with OGBG-PPA)
    python launch_experiment.py --config-file CONFIGS_CLDGN/config_LWF_Split_GraphSAGE_OGB.yml --splits-folder SPLITS/ --data-splits SPLITS/ogbg_ppa/ogbg_ppa_outer1_inner1.splits --data-root DATA/ --dataset-name ogbg_ppa --dataset-class data.dataset.OGBG --max-cpus 1 --max-gpus 1 --final-training-runs 5 --result-folder RESULTS_OGBG_PPA

To debug your code it is useful to add `--debug` to the command above. Notice, however, that the graphical interface will not work, as code will be executed sequentially. After debugging, if you need sequential execution, you can use `--max-cpus 1 --max-gpus 1 --gpus-per-task [0/1]` without the `--debug` option.  

## Troubleshooting
See the analogous section in the [PyDGN](https://github.com/diningphil/PyDGN) library.
