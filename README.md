# MMFormer: Multimodel Mamba Based GMMFormer for Efficient Partially Relevant Video Retrieval


## Catalogue <br> 
* [1. Getting Started](#getting-started)
* [2. Run](#run)
* [3. Results](#results)


## Getting Started

1\. Clone this repository:
```
git clone https://github.com/Lin-jingy/MamFusion.git
cd MamFusion
```

2\. Create a conda environment and install the dependencies:
```
conda create -n mamfusion-env python=3.10
conda activate mamfusion-env
pip install torch-2.4.1+cu118-cp310-cp310-linux_x86_64.whl
pip install torchaudio-2.4.1+cu118-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.19.1+cu118-cp310-cp310-linux_x86_64.whl
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
pip install mamba_ssm-2.2.1+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.3.0.post1+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -r requirements.txt
```

3\. Download Datasets: All features of TVR, ActivityNet Captions and Charades-STA are kindly provided by the authors of [MS-SL].


4\. Set root and data_root in config files (*e.g.*, ./Configs/tvr.py).

## Run

To train MamFusion on TVR:
```
cd src
python main.py -d tvr
```

To train GMMFormer on ActivityNet Captions:
```
cd src
python main.py -d act
```

To train GMMFormer on Charades-STA:
```
cd src
python main.py -d cha
```

## Results

### Quantitative Results

For this repository, the expected performance is:

| *Dataset* | *R@1* | *R@5* | *R@10* | *R@100* | *SumR* |
| ---- | ---- | ---- | ---- | ---- | ---- |
| TVR | 14.2 | 33.9 | 44.9 | 84.5 | 177.5 |
| ActivityNet Captions | 8.0 | 25.4 | 37.2 | 76.8 | 147.4 |
| Charades-STA | 2.0 | 8.8 | 14.2 | 51.5 | 76.5 |


[MS-SL]:https://github.com/HuiGuanLab/ms-sl





conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia