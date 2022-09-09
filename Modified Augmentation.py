import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
from zipfile import ZipFile

def augmentImage(filename):
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
    )

    save_here='Augmentated Images/Diseased/Rust'

    test_img=image.load_img('Guava Only Leaves/Diseased categorized/Rust Only Leaves'+'/'+filename)
    img=image.img_to_array(test_img)
    img=img.reshape((1,)+img.shape)

    i=0
    for batch in datagen.flow(img,batch_size=1,save_to_dir=save_here,save_prefix='test',save_format='jpeg'):
        i+=1
        if i>6:
            break


imageList=os.listdir('Guava Only Leaves/Diseased categorized/Rust Only Leaves')

for filename in imageList:
    augmentImage(filename)