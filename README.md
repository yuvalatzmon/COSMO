# Adaptive Confidence Smoothing for Generalized Zero-Shot Learning
Code for our paper: *Atzmon & Chechik, "Adaptive Confidence Smoothing for Generalized Zero-Shot Learning", CVPR 2019* <br>

<a href="https://arxiv.org/abs/1812.09903" target="_blank">paper</a> <br>
<a href="https://chechiklab.biu.ac.il/~yuvval/COSMO/" target="_blank">project page</a> <br>


## Installation
### Code and Data

 1. Download or clone the code in this repository.
 2. cd to the project directory
 3. Download the data (~500MB) by typing <br> `wget http://chechiklab.biu.ac.il/~yuvval/COSMO/data.zip`
 4. Extract the data by typing <br> `unzip -o data.zip`
 5. `data.zip` can be deleted
### Anaconda Environment

Quick installation under Anaconda:
  
    conda env create -f conda_requirements.yml
    
Alternatively, below are detailed installation instructions with Anaconda.<br>

    yes | conda create -n COSMO python=3.6
    
    conda activate COSMO

    yes | conda install pandas ipython jupyter nb_conda matplotlib 
    yes | conda install scikit-learn=0.19.1
    yes | pip install h5py seaborn
    
    # Other packages that I find useful, but are unrelated to this project
    yes | conda install -c conda-forge jupyter_contrib_nbextensions
    yes | conda install -c conda-forge jupyter_nbextensions_configurator
   


## Directory Structure
directory | file | description
---|---|---
`src/` | * | Sources files
`src/utils/` | * | Sources to useful utility procedures. 
`data/` | {CUB, AWA1, SUN, FLO}/ | Xian (CVPR, 2017) zero-shot data for CUB, AWA1, SUN, and FLOWER.
`data/LAGO_GZSL_predictions/*/` | pred_gzsl_val.npz | predictions of LAGO on GZSL-val set, when trained on train set.
`data/LAGO_GZSL_predictions/*/` | pred_gzsl_test.npz | predictions of LAGO on GZSL-test set, when trained on train+GZSLval set.
`data/XianGAN_predictions/*/` | pred_gzsl_val.npz | predictions of fCLSWGAN on GZSL-val set, when trained on train set.
`data/XianGAN_predictions/*/` | pred_gzsl_test.npz | predictions of fCLSWGAN on GZSL-test set, when trained on train+GZSLval set.
`output/` | * | Contains the outputs of the experimental framework (results & models). 
`output/seen_expert_model/` | * | Contains cached models for seen experts. 
`output/COSMO/` | * | Contains results of the experimental framework.



## Execute COSMO

To execute COSMO+LAGO, submit the following in the main project dir.

	PYTHONPATH="./" python src/main_cosmo.py --dataset_name=CUB --data_dir=data/CUB --zs_expert_name=LAGO
	PYTHONPATH="./" python src/main_cosmo.py --dataset_name=AWA1 --data_dir=data/AWA1 --zs_expert_name=LAGO
	PYTHONPATH="./" python src/main_cosmo.py --dataset_name=SUN --data_dir=data/SUN --zs_expert_name=LAGO
	 
To execute COSMO+fCLSWGAN, submit the following in the main project dir.

	PYTHONPATH="./" python src/main_cosmo.py --dataset_name=CUB --data_dir=data/CUB --zs_expert_name=XianGAN
	PYTHONPATH="./" python src/main_cosmo.py --dataset_name=AWA1 --data_dir=data/AWA1 --zs_expert_name=XianGAN
	PYTHONPATH="./" python src/main_cosmo.py --dataset_name=SUN --data_dir=data/SUN --zs_expert_name=XianGAN
	PYTHONPATH="./" python src/main_cosmo.py --dataset_name=FLO --data_dir=data/FLO --zs_expert_name=XianGAN

    
**NOTE:**

You **must** run the code from the project root directory.

## Cite our paper
If you use this code, please cite our paper.

    @inproceedings{atzmon2019COSMO,
    title={Adaptive Confidence Smoothing for Generalized Zero-Shot Learning},
    author={Atzmon, Yuval and Chechik, Gal},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year={2019},
    } 


