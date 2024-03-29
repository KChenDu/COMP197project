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

- Download the Cats-vs-Dogs dataset. The dataset in placed in the Onedrive folder `data`
  ```bash
  https://liveuclac-my.sharepoint.com/personal/ucabkc8_ucl_ac_uk/_layouts/15/onedrive.aspx?csf=1&web=1&e=eHOMTq&cid=b1ca2fff%2D7951%2D475e%2Dbffe%2Ddc90f122de3e&FolderCTID=0x012000125CE7251773044896CBF5C90A6BF230&id=%2Fpersonal%2Fucabkc8%5Fucl%5Fac%5Fuk%2FDocuments%2FCOMP197project%2Fdata%2Fkagglecatsanddogs%5F5340%2Ezip&parent=%2Fpersonal%2Fucabkc8%5Fucl%5Fac%5Fuk%2FDocuments%2FCOMP197project%2Fdata
  ```

- Unzip the dataset and place it under folder `data`

### ImageNet-1k-valid

- Download the ImageNet-1k-valid dataset. The dataset in placed in the Onedrive folder `data`
  ```bash
  https://liveuclac-my.sharepoint.com/:u:/r/personal/ucabkc8_ucl_ac_uk/Documents/COMP197project/data/ILSVRC2012_img_val.tar?csf=1&web=1&e=4YHG26
  ```

- Unzip the dataset and place it under folder `data`
