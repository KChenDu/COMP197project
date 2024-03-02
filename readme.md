# COMP197 Group Project
## Installation
The **most recommended** installation is to first create a conda environment and then run the following command:
- for CPU developer:
  ```bash
  conda install pytorch torchvision torchaudio cpuonly -c pytorch
  ```
- for CUDA developer, please check the CUDA version of your system:
  ```bash
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  ```
  or
  ```bash
  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  ```
Then, go to Anaconda and install `jupyter`, `h5py` and `matplotlib` manually (**recommended**) or run the following command:
```bash
conda install jupyter h5py matplotlib
```
***
## Datasets
- Oxford-IIIT Pet Dataset: Run the following command in project root directory:
  ```bash
  python data/download.py
  ```
