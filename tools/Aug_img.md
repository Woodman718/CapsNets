```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import os
from shutil import copy, rmtree 
import tensorflow as tf
# import cv2
```
lesion_type_dict = {
    'nv': 'Melanocytic_nevi',
    'mel': 'melanoma',
    'bkl': 'Benign_keratosis-like_lesions',
    'bcc': 'Basal_cell_carcinoma',
    'akiec': 'Actinic_keratoses',
    'vasc': 'Vascular_lesions',
    'df': 'Dermatofibroma'
}lesion_danger = {
    'nv': 0, # 0 for benign
    'mel': 1, # 1 for malignant
    'bkl': 0, # 0 for benign
    'bcc': 1, # 1 for malignant
    'akiec': 1, # 1 for malignant
    'vasc': 0,
    'df': 0
}

```python
# targetnames = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
# # To rename documents before action.
# # targetnames = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
# train_dir = "train50per/"
```


```python
def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)
```


```python
# source directory
cwd = os.getcwd()
data_root = os.path.abspath(os.path.join(cwd))
origin_data_path = os.path.join(data_root, "ISIC2019")
assert os.path.exists(origin_data_path), "path '{}' does not exist.".format(origin_data_path)
```


```python
data_class = [cla for cla in os.listdir(origin_data_path)
                if os.path.isdir(os.path.join(origin_data_path, cla))]
data_class
```




    ['vasc', 'nv', 'bkl', 'akiec', 'mel', 'df', 'bcc']




```python
!find ./ -type d -name '*point*'
```

    ./.ipynb_checkpoints



```python
!find ./ -type d -name '*point*' -exec rm -rf {} \;
```

    find: ‘./.ipynb_checkpoints’: 没有那个文件或目录



```python
# Augmentation directory
train_root = os.path.join(data_root,"t2019")
mk_file(train_root)
for cla in data_class:
    mk_file(os.path.join(train_root, cla))
!ls {train_root}
```

    akiec  bcc  bkl  df  mel  nv  vasc



```python
origin_data_path
```




    '/home/woodman/Jupyter/songbai/data/2019/ISIC2019'




```python
train_root
```




    '/home/woodman/Jupyter/songbai/data/2019/t2019'




```python
# Augmenting images and storing them in temporary directories 
for img_class in data_class:

    #creating temporary directories
    # creating a base directory
    aug_dir = "aug_dir"   
    # creating a subdirectory inside the base directory for images of the same class
    img_dir = os.path.join(aug_dir, 'img_dir')

    mk_file(img_dir)
    
    cla_path = os.path.join(origin_data_path,img_class)
    img_list = os.listdir(cla_path)

    # Copy images from the class train dir to the img_dir 
    for index, image in enumerate(img_list):
    # for file_name in img_list:

        # path of source image in training directory
        image_path = os.path.join(cla_path,image)
        # source = os.path.join(train_dir,img_class, file_name)

        # creating a target directory to send images 
        tag_path = os.path.join(data_root,img_dir,image)
        # target = os.path.join(img_dir, file_name)

        # copying the image from the source to target file
        copy(image_path, tag_path)

    # Temporary augumented dataset directory.
    source_path = os.path.join(data_root,aug_dir)

    # Augmented images will be saved to training directory
    save_path = os.path.join(train_root,img_class)
    # save_path = train_dir + img_class

    # Creating Image Data Generator to augment images
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(

        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'

    )

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(source_path,save_to_dir=save_path,
                                              save_format='jpg',save_prefix='trans_',
                                              target_size=(225, 300),batch_size=batch_size)

    # Generate the augmented images. Default:8000->51699
    if img_class == "nv":
        pass
    else:
        aug_images = 7000
        
        # num_files = len(img_list)
        # num_batches = int(np.ceil((aug_images - num_files) / batch_size))
        num_batches = int(np.ceil(aug_images / batch_size))

        # creating 8000 augmented images per class
        for i in range(0, num_batches):
            images, labels = next(aug_datagen)

        # delete temporary directory 
        rmtree(aug_dir)
```

    Found 253 images belonging to 1 classes.
    Found 12875 images belonging to 1 classes.
    Found 2624 images belonging to 1 classes.
    Found 867 images belonging to 1 classes.
    Found 4522 images belonging to 1 classes.
    Found 239 images belonging to 1 classes.
    Found 3323 images belonging to 1 classes.



```python
# copy origin_data_path to train_root().
total_num = 0
for cla in data_class:

    cla_path = os.path.join(origin_data_path, cla)
    images = os.listdir(cla_path)
    num = len(images)
    total_num += num
    for index, image in enumerate(images):
        image_path = os.path.join(cla_path, image)
        img_name = image_path.split('/')[-1].split(".")[0]
        savepath = os.path.join(train_root, cla,img_name + ".jpg")

        img = Image.open(image_path)
        img = img.resize((299, 299), resample=Image.LANCZOS)
        img.save(savepath,quality=100)
        # png
        # cv2.imwrite(savepath,img, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        # cv2.resize()
        # jpg
        # cv2.imwrite(savepath,img,[int(cv2.IMWRITE_JPEG_QUALITY),100])

        print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
    # break
    print()

print(f"processing {total_num} done!")
```

    [vasc] processing [228/228]
    [nv] processing [11588/11588]
    [bkl] processing [2362/2362]
    [akiec] processing [781/781]
    [mel] processing [4070/4070]
    [df] processing [216/216]
    [bcc] processing [2991/2991]
    [scc] processing [566/566]
    processing 22802 done!



```python
# detect 
total_num = 0
for cla in data_class:
    cla_path = os.path.join(train_root, cla)
    images = os.listdir(cla_path)
    num = len(images)
    total_num += num
    for index, image in enumerate(images):
 
        print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
    # break
    print()

print(f"processing {train_root.split('/')[-1]} : {total_num} done!")
```

    [vasc] processing [9527/9527]
    [nv] processing [11588/11588]
    [bkl] processing [9948/9948]
    [akiec] processing [9792/9792]
    [mel] processing [10000/10000]
    [df] processing [9840/9840]
    [bcc] processing [9993/9993]
    [scc] processing [9782/9782]
    processing train_2019 : 80470 done!



```python

```
