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
We will also use `smp` to import models and `loguru` to make a beautiful logging. One can intall it via pip:
```bash
pip install smp loguru
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