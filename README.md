# Skin Cancer Diagnosis with Capsule Networks README

## 1. Introduction
This repository focuses on a novel skin cancer assisted diagnosis method based on Capsule Networks with CBAM. The following sections detail the experimental setup, results, dataset, and how to use the code.

## 2. Experimental Equipment
The proposed method is implemented using PyTorch. All experiments were conducted on a tower workstation equipped with an Intel Core i5 - 11400KF and an NVIDIA GeForce RTX 3070.

## 3. Environment Setup
To run the code, you need to set up the following environment:
### 3.1 Create a Conda Environment
```bash
# Create a new Conda environment named 'pytorch - gpu' with Python 3.10 and specific versions of PyTorch and related libraries
# The channels '-c pytorch - c nvidia' are used to fetch packages
conda create -n pytorch-gpu python = 3.10 pytorch == 1.13 torchvision==0.14.1 torchaudio==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
### 3.2 Install Additional Dependencies
```bash
# Install essential Python libraries for data handling, visualization, and neural network - related operations
conda install numpy seaborn prettytable matplotlib tqdm pillow scikit-learn ipykernel
```
### 3.3 Install Extra Python Packages via Pip
```bash
# Install packages for model size and FLOPs calculation, tensor operations, and model summary
pip install thop einops torch-summary
```
### 3.4 Activate the Environment and Install Kernel (Optional for Jupyter)
```bash
# Activate the 'pytorch-gpu' environment
conda activate pytorch-gpu
# Install the kernel for the 'pytorch-gpu' environment in Jupyter
python -m ipykernel install --user --name pytorch-gpu --display-name "pytorch-gpu"
```

### 3.5 Environment Verification
You can verify the installation with the following Python code:
```python
import torch
import torchvision
import torchaudio
print('Pytorch version\t:', torch.__version__)
print('Torchvision \t:', torchvision.__version__)
print('Torchaudio \t:', torchaudio.__version__)
print('CUDA version\t:', torch.version.cuda)
print('CUDA status\t:', torch.cuda.is_available())
print('GPU\t\t:', torch.cuda.get_device_name())
```
**Sample Output**:
```
# pytorch - gpu #
Pytorch version	: 1.13.1+cu117
Torchvision 	: 0.14.1+cu117
Torchaudio 	: 0.13.1+cu117
CUDA version	: 11.7
CUDA status	: True
GPU		: NVIDIA GeForce RTX 3070
```

## 4. Results
### 4.1 Evaluation metrics on the HAM10000 (Augment)
<table> 
 <tr><th>Evaluation metrics</th><th>Comparison with other methods</th></tr> 
<tr><td> 


|  Type  | Precision | Recall |   F1  | Accuracy |
|:--------:|:-------:|:-------------:|:--------:|:----------:|
| akiec  |   0.992   | 0.996  | 0.994  |          |
|  bcc   |   0.9896  | 0.9961 | 0.9929 |          |
|  bkl   |   0.9934  | 0.9882 | 0.9908 |          |
|   df   |   0.9981  |  1.0   | 0.9991 |          |
|  mel   |   0.9858  | 0.9897 | 0.9877 |          |
|   nv   |    1.0    | 0.9881 | 0.994  |          |
|  vasc  |    1.0    |  1.0   |  1.0   |          |
| overall:|   0.9941  | 0.994  | 0.9941 |  0.9937  |

</td><td>

|Method |Accuracy [%] |Params(M)  |FLOPs(G)|
|:--------:|:-------------:|:-------------:|:-------------:|
Inception V3  |92.10  |22.80 |5.73
ResNet 50 |92.31 |25.60 |4.10
DenseNet-201 |92.87  |20.01|4.28
IRv2-SA |93.47  |47.5 |25.46
IM-CNN  |95.10 |-|-
Proposed(Ours) |99.37  |1.41  |2.74

</td></tr> </table>

### 4.2 Evaluation metrics on the HAM10000
#### 4.2.1 Evaluation metrics and LCK
<table> 
 <tr><th>Evaluation metrics</th><th> LKC(large-kernel convolution)</th></tr> 
<tr><td> 

|  Type  | Precision | Recall |   F1  | Accuracy |
|:--------:|:-------:|:-------------:|:--------:|:----------:|
| akiec  |    1.0    | 0.9394 | 0.9687 |          |
|  bcc   |   0.8983  |  1.0   | 0.9464 |          |
|  bkl   |   0.9444  | 0.9027 | 0.9231 |          |
|   df   |    0.8    | 0.7273 | 0.7619 |          |
|  mel   |   0.8872  |  1.0   | 0.9402 |          |
|   nv   |   0.9954  | 0.9714 | 0.9832 |          |
|  vasc  |   0.8235  |  1.0   | 0.9032 |          |
| overall:|   0.907   | 0.9344 | 0.9181 |  0.9652  |

</td><td>

![LKC](https://github.com/Woodman718/CapsNets/blob/main/Images/LKC.png#pic_center)

</td></tr> </table>

**Note**: LKC with different kernel sizes. The N of the label ”kernel - N” indicates the size of the convolution kernel. For instance, kernel-21 means using an LKC with a 21×21 convolution kernel.

#### 4.2.2 Attention
![Atten](https://github.com/Woodman718/CapsNets/blob/main/Images/Attention.png#pic_center)

#### 4.2.3 Generalization Performance
**Dataset**:  https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
The COVID-19 Radiography Database consisted of 21165 images. Among them, covid(3616), normal(10192), opacity(6012), viral(1345).

<table> 
<tr><th>Evaluation Metrics</th><th>Distribution of the COVID-19 Radiography Dataset</th></tr> 
<tr><td> 

|  Type  | Precision | Recall |  F1  | Accuracy |
|:--------:|:-------------:|:-------------:|:--------:|:----------:|
|  covid  |   0.9972  |  1.0   | 0.999 |          |
|  normal |   0.999   | 0.996  | 0.998 |          |
| opacity |   0.995   | 0.997  | 0.996 |          |
|  viral  |   0.9926  |  1.0   | 0.996 |          |
|  overall: |           |        |       |  0.9972  |

</td><td>

 ![dis_data](https://github.com/Woodman718/CapsNets/blob/main/Images/Dis_COVID-19_data.png)

</td></tr>
</table>

**Source Data**: http://dx.doi.org/10.5281/zenodo.1214456
Jakob Nikolas Kather, Johannes Krisam, et al., "Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study," PLOS Medicine, vol. 16, no. 1, pp. 1–22, 01 2019.
This is a slightly different version of the "NCT - CRC - HE - 100K" image set: This set contains 100,000 images in 9 tissue classes at 0.5 MPP and was created from the same raw data as "NCT - CRC - HE - 100K". However, no color normalization was applied to these images. Consequently, staining intensity and color slightly varies between the images. Please note that although this image set was created from the same data as "NCT - CRC - HE - 100K", the image regions are not completely identical because the selection of non - overlapping tiles from raw images was a stochastic process.

<table> 
<tr><th>Evaluation Metrics</th><th>NCT-CRC-HE-100K-NONORM</th></tr> 
<tr><td> 



|  Type  | Precision | Recall |  F1  | Accuracy |
|:---------|:-------------:|:-------------:|:--------:|:----------:|
|  ADI   |    1.0    |  1.0   |  1.0  |          |
|  BACK  |    1.0    |  1.0   |  1.0  |          |
|  DEB   |    1.0    |  1.0   |  1.0  |          |
|  LYM   |    1.0    | 0.998  | 0.999 |          |
|  MUC   |   0.9978  | 0.998  | 0.998 |          |
|  MUS   |   0.9985  | 0.999  | 0.999 |          |
|  NORM  |   0.9989  |  1.0   | 0.999 |          |
|  STR   |   0.999   | 0.997  | 0.998 |          |
|  TUM   |   0.9979  | 0.999  | 0.999 |          |
| overall: |           |        |       |  0.9991  |

</td><td>

 ![dis_data](https://github.com/Woodman718/CapsNets/blob/main/Images/Dis_NCT-CRC-HE-100K-NONORM.png)

</td></tr>
</table>)

## 5. Dataset
![Data](https://github.com/Woodman718/CapsNets/blob/main/Images/Aug-Dis.png)

The distribution of the seven disease types before and after data augmentation. In the clusters of bars with the same color, the left bar represents the sample distribution after data augmentation, while the right bar represents the initial distribution of the dataset.

**Example of Skin lesions in HAM10000 dataset**: Among them, BKL, DF, NV, and VASC are benign tumors, whereas AKIEC, BCC, and MEL are malignant tumors.

**Available at**:
- https://challenge.isic-archive.com/data/#2018
- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- https://aistudio.baidu.com/aistudio/datasetdetail/218024 (ours)

**Citation**:
P. Tschandl, C. Rosendahl, and H. Kittler, “The ham10000 dataset,a large collection of multi - source dermatoscopic images of common pigmented skin lesions,” Scientific data, vol. 5, no. 1, pp. 1–9, 2018.

## 6. License
The dataset is released under a Creative Commons Attribution 4.0 License. For more information, see https://creativecommons.org/licenses/by/4.0/.

## 7. Related Work
### 7.1
```
@article{WangIMCC,
  author={Wang, Sutong and Yin, Yunqiang and Wang, Dujuan and Wang, Yanzhang and Jin, Yaochu},
  journal={IEEE Transactions on Cybernetics}, 
  title={Interpretability - Based Multimodal Convolutional Neural Networks for Skin Lesion Diagnosis}, 
  year={2022},
  volume={52},
  number={12},
  pages={12623 - 12637},
  doi={10.1109/TCYB.2021.3069920}
}
```
### 7.2
```
@article{xia2017exploring,
  title={Exploring Web images to enhance skin disease analysis under a computer vision framework},
  author={Xia, Yingjie and Zhang, Luming and Meng, Lei and Yan, Yan and Nie, Liqiang and Li, Xuelong},
  journal={IEEE Transactions on Cybernetics},
  volume={48},
  number={11},
  pages={3080--3091},
  year={2017},
  publisher={IEEE}
}
```

## 8. Citation
If you use our method for your research or application, please consider citation:
### 8.1
```
@ARTICLE{CapsNets,
  author={Lan, Zhangli and Cai, Songbai and Zhu, Jiqiang and Xu, Yuantong},
  journal={XXX on XXX}, 
  title={A Novel Skin Cancer Assisted Diagnosis Method based on Capsule Networks with CBAM}, 
  year={},
  volume={},
  number={},
  pages={},
  doi={10.36227/techrxiv.23291003},
}
```
### 8.2
```
@ARTICLE{9791221,
  author={Lan, Zhangli and Cai, Songbai and He, Xu and Wen, Xinpeng},
  journal={IEEE Access}, 
  title={FixCaps: An Improved Capsules Network for Diagnosis of Skin Cancer}, 
  year={2022},
  volume={10},
  number={},
  pages={76261 - 76267},
  doi={10.1109/ACCESS.2022.3181225}
}
```
