# COMP197 Group Project
## Installation
The **most recommended** installation is to first create a conda environment and then run the following command:
- for CPU developer:
  ```bash
  conda install pytorch torchvision cpuonly -c pytorch
  ```
- for CUDA developer, please check the CUDA version of your system:
  ```bash
  conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
  ```
  or
  ```bash
  conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
  ```
Then, go to Anaconda and install each package in `requirements.txt` manually (**recommended**) or run the following command:
```bash
conda install --file requirements.txt
```
### Extra
We will also use `segmentation-models-pytorch` to import models and `loguru` to make a beautiful logging. One can intall it via pip:
```bash
pip install segmentation-models-pytorch loguru
```
***
## Datasets
### Oxford-IIIT Pet Dataset

Run the following command in project root directory:
```bash
python data/download.py
 ```
then the dataset will appear in `data/oxford-iiit-pet`.
- **Caution**: Do not push the datasets to code repository. Manage them in [team's OneDrive](https://liveuclac-my.sharepoint.com/:f:/r/personal/ucabkc8_ucl_ac_uk/Documents/COMP197project?csf=1&web=1&e=eHOMTq).

### Cats-vs-Dogs Dataset

[Download](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabkc8_ucl_ac_uk/ERj5dgFcEhxOlZ-k6nvaynUBgWZKVTej3dkD1T2529KTUA?e=CjmJWl) `kagglecatsanddogs_5340.zip`, unzip it and place `PetImages` folder in `data` directory.

### ImageNet-1k-valid

[Download](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabkc8_ucl_ac_uk/EtmYnd5lrbpDq7DJmxivmRQBTnyuzem_Avu2T4p7glxcrA?e=84ciAG) `ILSVRC2012_devkit_t12.tar.gz` and `ILSVRC2012_img_val.tar.gz`, then place them in `data` directory.
