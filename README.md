## Abstract
The diagnosis of skin lesions is crucial for early screening of skin cancer, requiring high accuracy and stability. Although deep learning-based methods have made significant progress in skin lesion diagnosis, their stability and generalization still remain a challenge. Current methods tend to generate redundant features when dealing with skin lesions with intraclass diversity and inter-class similarity, leading to poor performance in clinical applications. Therefore, this paper proposes a highly available Capsule Network for skin lesion diagnosis. Firstly, a high-performance LKC with a kernel size of 21×21 is applied at the initial layer to achieve a larger receptive field than CapsNets. To further improve diagnostic accuracy and computational efficiency, we introduce an asymmetric convolutional group and global max pooling. Additionally, a convolutional attention module is employed to mitigate the loss of spatial information in the network. Moreover, to obtain a more comprehensive representation of the output, we convert the network output from a vector to a matrix and measure the output matrix using kernel norm. Finally, by optimizing the CBAM, Squash function, Margin Loss function, and Dynamic Routing, the model can meet a wider range of pathological evaluation requirements. The experimental results show that the proposed algorithm achieves a skin lesion diagnosis accuracy of 99.37% on the HAM10000 dataset, and an accuracy of 99.72% and 99.91% on the COVID-19 and NCT-CRC-HE datasets, which are used for the diagnosis of COVID-19 and colorectal cancer, respectively. Our proposed algorithm not only achieves accurate diagnosis of skin lesions, but also demonstrates remarkable generalization and stability. Its performance on all three datasets surpasses that of the current state-of-the-art methods, providing a reliable and efficient solution for diagnosis and treatment in the medical field.

## Results
1. Evaluation metrics on the HAM10000 (Augment).
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

2  Evaluation metrics on the HAM10000.
1) Evaluation metrics and LCK
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

```
LKC with different kernel sizes.  The N of the label ”kernel-N” indicates the size of the convolution kernel.  For instance, kernel-21 means using an LKC with a 21×21 convolution kern
```

2). Attention.

![Atten]https://github.com/Woodman718/CapsNets/blob/main/Images/Attention#pic_center)

3 Generalization Performance

```
Dataset:  https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
The COVID-19 Radiography Database consisted of 21165 images.
Among them, covid(3616),normal(10192),opacity(6012),viral(1345).
```

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

```
Source Data: http://dx.doi.org/10.5281/zenodo.1214456
Jakob Nikolas Kather, Johannes Krisam, et al., "Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study," PLOS Medicine, vol. 16, no. 1, pp. 1–22, 01 2019.
This is a slightly different version of the "NCT-CRC-HE-100K" image set: This set contains 100,000 images in 9 tissue classes at 0.5 MPP and was created from the same raw data as "NCT-CRC-HE-100K". 
However, no color normalization was applied to these images. Consequently, staining intensity and color slightly varies between the images. Please note that although this image set was created from the same data as "NCT-CRC-HE-100K", the image regions are not completely identical because the selection of non-overlapping tiles from raw images was a stochastic process.
```

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
</table>

## Dataset

![Data](https://github.com/Woodman718/CapsNets/blob/main/Images/Aug-Dis.png)

The distribution of the seven disease types before and after data augmentation. In the clusters of bars with the same color, the left bar represents the sample distribution after data augmentation, while the right bar represents the initial distribution of the dataset.

```
Example of Skin lesions in HAM10000 dataset.
Among them, BKL, DF, NV, and VASC are benign tumors, whereas AKIEC, BCC, and MEL are malignant tumors.

Available:
https://challenge.isic-archive.com/data/#2018
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
https://aistudio.baidu.com/aistudio/datasetdetail/218024 (ours)
```

HAM10000 dataset:

```
Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)", 2018; https://arxiv.org/abs/1902.03368

Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018). 
Available: https://www.nature.com/articles/sdata2018161, https://arxiv.org/abs/1803.10417
```

## License

The dataset is released under a Creative Commons Attribution 4.0 License.
For more information, see https://creativecommons.org/licenses/by/4.0/ .

## Related Work

a. 

```
@article{WangIMCC,
  author={Wang, Sutong and Yin, Yunqiang and Wang, Dujuan and Wang, Yanzhang and Jin, Yaochu},
  journal={IEEE Transactions on Cybernetics}, 
  title={Interpretability-Based Multimodal Convolutional Neural Networks for Skin Lesion Diagnosis}, 
  year={2022},
  volume={52},
  number={12},
  pages={12623-12637},
  doi={10.1109/TCYB.2021.3069920}
}
```
b
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

## Citation

If you use our method for your research or aplication, please consider citation:

```
To be continued
```
