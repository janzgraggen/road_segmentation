# Road segmentation from aerial imagery with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#examples">Examples</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About
This poject aims to to evaluate the performance of different model architectures and applying different image transforms for road segmentation.

## Installation

Follow these steps:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## Submission

To reproduce the same submission we got on AICrowed use the following command:

```bash
python3 submit.py
```

> [!NOTE]  
> To use the pre-trained weights you should have [git lfs](https://git-lfs.com/) installed.

This will create a `submission.csv` file in the root of this repo.

To train the same model which was used for the submission use:

```bash
python3 train.py -cn exp_arch_mobilenetv3
```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference on the test dataset and generate submission file:
```bash
python3 submit.py HYDRA_CONFIG_ARGUMENTS
```

To visualize the results in the submission file:
```bash
python3 visualize.py -i IMAGE_INDEX 
```

## Examples

To train the baseline configuration 

```bash
python3 train.py -cn baseline
```

To run inference on the test dataset and generate the submission file:
```bash
python3 submit.py -cn submit
```

To run inference on the validation dataset and generate the submission file:
```bash
python3 submit.py -cn validation 
```

To visualize the inference on the first image (a submission file must already be generated):
```bash
python3 visualize.py -i 1 
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
