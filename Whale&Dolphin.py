# %% [code]
# %% [markdown] {"papermill":{"duration":0.021691,"end_time":"2022-02-06T11:38:55.249528","exception":false,"start_time":"2022-02-06T11:38:55.227837","status":"completed"},"tags":[]}
# # CNN with Keras Stater

# %% [markdown] {"papermill":{"duration":0.013788,"end_time":"2022-02-06T11:38:55.357426","exception":false,"start_time":"2022-02-06T11:38:55.343638","status":"completed"},"tags":[]}
# ### Importing Libraries

# %% [code] {"papermill":{"duration":5.774181,"end_time":"2022-02-06T11:39:01.145537","exception":false,"start_time":"2022-02-06T11:38:55.371356","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-04-14T05:18:35.485446Z","iopub.execute_input":"2022-04-14T05:18:35.486121Z","iopub.status.idle":"2022-04-14T05:18:40.856173Z","shell.execute_reply.started":"2022-04-14T05:18:35.486012Z","shell.execute_reply":"2022-04-14T05:18:40.855424Z"}}
import numpy as np 
import pandas as pd 
import os
import gc
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
from tqdm.autonotebook import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import keras.backend as K
from keras.models import Sequential
from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, LocallyConnected2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %% [code] {"execution":{"iopub.status.busy":"2022-04-14T05:18:40.857741Z","iopub.execute_input":"2022-04-14T05:18:40.857996Z","iopub.status.idle":"2022-04-14T05:18:41.166163Z","shell.execute_reply.started":"2022-04-14T05:18:40.857959Z","shell.execute_reply":"2022-04-14T05:18:41.165454Z"}}

from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# VGG16
from keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models
from keras.models import Model
from keras.layers import Input
from keras import backend as K
from keras import optimizers

import numpy as np
import argparse
import cv2
import os, os.path


# %% [code] {"execution":{"iopub.status.busy":"2022-04-11T16:24:13.138037Z","iopub.execute_input":"2022-04-11T16:24:13.138614Z","iopub.status.idle":"2022-04-11T16:24:13.143731Z","shell.execute_reply.started":"2022-04-11T16:24:13.138576Z","shell.execute_reply":"2022-04-11T16:24:13.142813Z"}}
train_path = '../input/whale2-cropped-dataset/cropped_train_images/cropped_train_images'
# cat and dog images in subfolders 'train/cats' and 'train/dogs' 

# path to validation images
validate_path = 'validate_ffd'
# cat and dog images in subfolders 'validate/cats' and 'validate/dogs' 

# images to be resized to (image_dim) x (image_dim)
image_dim = 128


# ---- define data generators ----
batch_size = 32

# %% [code] {"execution":{"iopub.status.busy":"2022-04-13T18:19:44.032922Z","iopub.execute_input":"2022-04-13T18:19:44.033716Z","iopub.status.idle":"2022-04-13T18:19:44.040301Z","shell.execute_reply.started":"2022-04-13T18:19:44.033652Z","shell.execute_reply":"2022-04-13T18:19:44.038411Z"}}
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

# %% [code] {"papermill":{"duration":0.129922,"end_time":"2022-02-06T11:39:01.290815","exception":false,"start_time":"2022-02-06T11:39:01.160893","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-04-14T05:18:41.167534Z","iopub.execute_input":"2022-04-14T05:18:41.167791Z","iopub.status.idle":"2022-04-14T05:18:41.301651Z","shell.execute_reply.started":"2022-04-14T05:18:41.167757Z","shell.execute_reply":"2022-04-14T05:18:41.300799Z"}}
train_df = pd.read_csv("../input/whale2-cropped-dataset/train2.csv")
#train_df=train_df.drop_duplicates(subset=['individual_id'],keep='last')
train_df.head()

# %% [code] {"papermill":{"duration":0.022382,"end_time":"2022-02-06T11:39:01.328361","exception":false,"start_time":"2022-02-06T11:39:01.305979","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-04-14T05:18:41.562582Z","iopub.execute_input":"2022-04-14T05:18:41.563106Z","iopub.status.idle":"2022-04-14T05:18:41.568738Z","shell.execute_reply.started":"2022-04-14T05:18:41.563068Z","shell.execute_reply":"2022-04-14T05:18:41.567945Z"}}
train_df.shape

# %% [code] {"execution":{"iopub.status.busy":"2022-04-14T05:18:41.857833Z","iopub.execute_input":"2022-04-14T05:18:41.858469Z","iopub.status.idle":"2022-04-14T05:18:41.881759Z","shell.execute_reply.started":"2022-04-14T05:18:41.858433Z","shell.execute_reply":"2022-04-14T05:18:41.880845Z"}}
print('Species Count: ',len(train_df['species'].value_counts()))
train_df['species'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2022-04-14T05:18:44.786953Z","iopub.execute_input":"2022-04-14T05:18:44.787257Z","iopub.status.idle":"2022-04-14T05:18:44.817610Z","shell.execute_reply.started":"2022-04-14T05:18:44.787220Z","shell.execute_reply":"2022-04-14T05:18:44.816988Z"}}
train_df['frequency'] = train_df['species'].map(train_df['species'].value_counts())
train_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-04-14T05:18:45.076091Z","iopub.execute_input":"2022-04-14T05:18:45.076490Z","iopub.status.idle":"2022-04-14T05:18:45.136798Z","shell.execute_reply.started":"2022-04-14T05:18:45.076457Z","shell.execute_reply":"2022-04-14T05:18:45.136112Z"}}
print('Before fixing duplicate labels : ')
print("Number of unique species : ", train_df['species'].nunique())

train_df['species'].replace({
    'bottlenose_dolpin' : 'bottlenose_dolphin',
    'kiler_whale' : 'killer_whale',
    'beluga' : 'beluga_whale',
    'globis' : 'short_finned_pilot_whale',
    'pilot_whale' : 'short_finned_pilot_whale'
},inplace =True)

print('\nAfter fixing duplicate labels : ')
print("Number of unique species : ", train_df['species'].nunique())


train_df['class'] = train_df['species'].apply(lambda x: x.split('_')[-1])
train_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-04-14T05:18:48.199522Z","iopub.execute_input":"2022-04-14T05:18:48.200031Z","iopub.status.idle":"2022-04-14T05:18:48.220655Z","shell.execute_reply.started":"2022-04-14T05:18:48.199995Z","shell.execute_reply":"2022-04-14T05:18:48.219954Z"}}
print('Species Count: ',len(train_df['species'].value_counts()))
train_df['species'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2022-04-14T05:18:49.840821Z","iopub.execute_input":"2022-04-14T05:18:49.841527Z","iopub.status.idle":"2022-04-14T05:18:49.875464Z","shell.execute_reply.started":"2022-04-14T05:18:49.841485Z","shell.execute_reply":"2022-04-14T05:18:49.874349Z"}}
train_df_species = train_df[train_df["species"] == 'false_killer_whale']
train_df_species

# %% [code] {"execution":{"iopub.status.busy":"2022-04-14T05:19:39.212479Z","iopub.execute_input":"2022-04-14T05:19:39.212740Z","iopub.status.idle":"2022-04-14T05:19:39.229543Z","shell.execute_reply.started":"2022-04-14T05:19:39.212709Z","shell.execute_reply":"2022-04-14T05:19:39.228790Z"}}
y, label_encoder = prepare_labels(train_df_species['individual_id'])
y.shape

# %% [code] {"execution":{"iopub.status.busy":"2022-04-11T06:53:25.060255Z","iopub.execute_input":"2022-04-11T06:53:25.060768Z","iopub.status.idle":"2022-04-11T06:53:25.142975Z","shell.execute_reply.started":"2022-04-11T06:53:25.06073Z","shell.execute_reply":"2022-04-11T06:53:25.142083Z"}}
import os
from PIL import Image
from PIL import ImageFilter
filelist = train_df['image'].loc[(train_df['frequency']<1000)].tolist()
for count in range(0,2):
  
  for imagefile in filelist:
    os.chdir('../input/whale2-cropped-dataset/cropped_train_images/cropped_train_images')
    im=Image.open(imagefile)
    im=im.convert("RGB")
    r,g,b=im.split()
    r=r.convert("RGB")
    g=g.convert("RGB")
    b=b.convert("RGB")
    im_blur=im.filter(ImageFilter.GaussianBlur)
    im_unsharp=im.filter(ImageFilter.UnsharpMask)
    
    os.chdir('../input/whale2-cropped-dataset/cropped_train_images/cropped_train_images_copy')
    r.save(str(count)+'r_'+imagefile)
    g.save(str(count)+'g_'+imagefile)
    b.save(str(count)+'b_'+imagefile)
    im_blur.save(str(count)+'bl_'+imagefile)
    im_unsharp.save(str(count)+'un_'+imagefile)

# %% [code] {"execution":{"iopub.status.busy":"2022-04-13T18:20:15.551621Z","iopub.execute_input":"2022-04-13T18:20:15.551947Z","iopub.status.idle":"2022-04-13T18:22:31.077243Z","shell.execute_reply.started":"2022-04-13T18:20:15.551889Z","shell.execute_reply":"2022-04-13T18:22:31.076202Z"}}
train_generator=datagen.flow_from_dataframe(
dataframe=train_df,
directory="../input/whale2-cropped-dataset/cropped_train_images/cropped_train_images/",
x_col="image",
y_col="individual_id",
subset="training",
batch_size=512,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(128,128))

valid_generator=datagen.flow_from_dataframe(
dataframe=train_df,
directory="../input/whale2-cropped-dataset/cropped_train_images/cropped_train_images/",
x_col="image",
y_col="individual_id",
subset="validation",
batch_size=512,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(128,128))

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.674954Z","iopub.execute_input":"2022-03-27T07:32:52.675587Z","iopub.status.idle":"2022-03-27T07:32:52.679479Z","shell.execute_reply.started":"2022-03-27T07:32:52.675534Z","shell.execute_reply":"2022-03-27T07:32:52.678756Z"}}
# train_df['count'] = train_df.groupby('individual_id',as_index=False)['individual_id'].transform(lambda x: x.count())
# train_df.head(10)

# %% [markdown] {"papermill":{"duration":0.014765,"end_time":"2022-02-06T11:39:01.358066","exception":false,"start_time":"2022-02-06T11:39:01.343301","status":"completed"},"tags":[]}
# ## Functions

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.681532Z","iopub.execute_input":"2022-03-27T07:32:52.682317Z","iopub.status.idle":"2022-03-27T07:32:52.689286Z","shell.execute_reply.started":"2022-03-27T07:32:52.682279Z","shell.execute_reply":"2022-03-27T07:32:52.688584Z"},"jupyter":{"source_hidden":true}}
# import random
# random.seed(0)
# IMG_HEIGHT = 32
# IMG_WIDTH = 32
# def Loading_Images(data, m, dataset):
#     print("Loading images")
#     n = int(m/10)
#     X_train = []
#     for i in tqdm(range(n)):
#         fig = random.sample(list(data['image']),1)
#         image= np.array(Image.open("../input/happy-whale-and-dolphin/"+dataset+"/"+fig[0]))
#         image= np.resize(image,(IMG_HEIGHT,IMG_WIDTH,3))
#         image = image.astype('float32')
#         image /= 255  
#         X_train.append(image)
#     return X_train



# %% [code] {"papermill":{"duration":0.024637,"end_time":"2022-02-06T11:39:01.397649","exception":false,"start_time":"2022-02-06T11:39:01.373012","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-04-14T05:18:55.461587Z","iopub.execute_input":"2022-04-14T05:18:55.462103Z","iopub.status.idle":"2022-04-14T05:18:55.469155Z","shell.execute_reply.started":"2022-04-14T05:18:55.462063Z","shell.execute_reply":"2022-04-14T05:18:55.468428Z"}}
def Loading_Images(data, m, dataset):
    print("Loading images")
    X_train = np.zeros((m, 200, 200, 3))
    count = 0
    for fig in tqdm(data['image']):
        img = image.load_img("../input/whale2-cropped-dataset/"+dataset+"/"+fig, target_size=(200, 200, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        X_train[count] = x
        count +=  1
    return X_train

def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    y = onehot_encoded
    return y, label_encoder

# %% [code] {"papermill":{"duration":4522.329831,"end_time":"2022-02-06T12:54:23.742277","exception":false,"start_time":"2022-02-06T11:39:01.412446","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-04-14T05:18:57.918212Z","iopub.execute_input":"2022-04-14T05:18:57.918656Z","iopub.status.idle":"2022-04-14T05:19:34.929221Z","shell.execute_reply.started":"2022-04-14T05:18:57.918618Z","shell.execute_reply":"2022-04-14T05:19:34.928432Z"}}
X = Loading_Images(train_df_species,3326, "cropped_train_images/cropped_train_images")
X/=255

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.71134Z","iopub.execute_input":"2022-03-27T07:32:52.71212Z","iopub.status.idle":"2022-03-27T07:32:52.718889Z","shell.execute_reply.started":"2022-03-27T07:32:52.712063Z","shell.execute_reply":"2022-03-27T07:32:52.718131Z"}}
# np.savez("X_51033", X)

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.720328Z","iopub.execute_input":"2022-03-27T07:32:52.720816Z","iopub.status.idle":"2022-03-27T07:32:52.727462Z","shell.execute_reply.started":"2022-03-27T07:32:52.720777Z","shell.execute_reply":"2022-03-27T07:32:52.72657Z"}}
# y = train_df['species']
# y = np.array(y)
# y



# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.72891Z","iopub.execute_input":"2022-03-27T07:32:52.72936Z","iopub.status.idle":"2022-03-27T07:32:52.735936Z","shell.execute_reply.started":"2022-03-27T07:32:52.729324Z","shell.execute_reply":"2022-03-27T07:32:52.735056Z"}}
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y, num_classes=26)

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.737107Z","iopub.execute_input":"2022-03-27T07:32:52.737729Z","iopub.status.idle":"2022-03-27T07:32:52.744939Z","shell.execute_reply.started":"2022-03-27T07:32:52.737692Z","shell.execute_reply":"2022-03-27T07:32:52.744206Z"}}
# le = LabelEncoder()
# y = le.fit_transform(y)

# %% [code] {"papermill":{"duration":0.310577,"end_time":"2022-02-06T12:54:24.068919","exception":false,"start_time":"2022-02-06T12:54:23.758342","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-04-14T04:49:11.986916Z","iopub.execute_input":"2022-04-14T04:49:11.987457Z","iopub.status.idle":"2022-04-14T04:49:12.034143Z","shell.execute_reply.started":"2022-04-14T04:49:11.987418Z","shell.execute_reply":"2022-04-14T04:49:12.033353Z"}}
y, label_encoder = prepare_labels(train_df_species['individual_id'])
y.shape

# %% [code] {"execution":{"iopub.status.busy":"2022-03-29T14:25:26.288115Z","iopub.execute_input":"2022-03-29T14:25:26.288366Z","iopub.status.idle":"2022-03-29T14:25:26.436239Z","shell.execute_reply.started":"2022-03-29T14:25:26.288337Z","shell.execute_reply":"2022-03-29T14:25:26.435605Z"},"jupyter":{"outputs_hidden":true}}
# a = image.load_img("../input/whale2-cropped-dataset/cropped_train_images/cropped_train_images/000c476c11bad5.jpg", target_size=(500, 500, 3))
# a

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.772072Z","iopub.execute_input":"2022-03-27T07:32:52.772598Z","iopub.status.idle":"2022-03-27T07:32:52.781031Z","shell.execute_reply.started":"2022-03-27T07:32:52.772543Z","shell.execute_reply":"2022-03-27T07:32:52.78018Z"}}
# b = image.load_img("../input/whale2-cropped-dataset/cropped_train_images/cropped_train_images/00021adfb725ed.jpg", target_size=(128, 128, 3))
# b

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.783613Z","iopub.execute_input":"2022-03-27T07:32:52.784124Z","iopub.status.idle":"2022-03-27T07:32:52.789256Z","shell.execute_reply.started":"2022-03-27T07:32:52.784088Z","shell.execute_reply":"2022-03-27T07:32:52.788654Z"}}
# X = Loading_Images(train_df, train_df.shape[0], "cropped_train_images/cropped_train_images")

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.790529Z","iopub.execute_input":"2022-03-27T07:32:52.791114Z","iopub.status.idle":"2022-03-27T07:32:52.797323Z","shell.execute_reply.started":"2022-03-27T07:32:52.791015Z","shell.execute_reply":"2022-03-27T07:32:52.796664Z"}}
# np.savez("X_detic_crop_64", X)

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.798775Z","iopub.execute_input":"2022-03-27T07:32:52.799372Z","iopub.status.idle":"2022-03-27T07:32:52.804761Z","shell.execute_reply.started":"2022-03-27T07:32:52.799276Z","shell.execute_reply":"2022-03-27T07:32:52.804021Z"}}
# !ls -1

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.806089Z","iopub.execute_input":"2022-03-27T07:32:52.806726Z","iopub.status.idle":"2022-03-27T07:32:52.812242Z","shell.execute_reply.started":"2022-03-27T07:32:52.806632Z","shell.execute_reply":"2022-03-27T07:32:52.811492Z"}}
# y = np.expand_dims(y,axis=0)

# %% [code] {"papermill":{"duration":0.168833,"end_time":"2022-02-06T12:54:24.253731","exception":false,"start_time":"2022-02-06T12:54:24.084898","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-03-27T07:32:52.814586Z","iopub.execute_input":"2022-03-27T07:32:52.816696Z","iopub.status.idle":"2022-03-27T07:32:52.820525Z","shell.execute_reply.started":"2022-03-27T07:32:52.816661Z","shell.execute_reply":"2022-03-27T07:32:52.819807Z"}}
# y =  np.array(y_train)
# y.shape
# y

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.822183Z","iopub.execute_input":"2022-03-27T07:32:52.822752Z","iopub.status.idle":"2022-03-27T07:32:52.827735Z","shell.execute_reply.started":"2022-03-27T07:32:52.822692Z","shell.execute_reply":"2022-03-27T07:32:52.827018Z"}}
# y = np.transpose(y)

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.829094Z","iopub.execute_input":"2022-03-27T07:32:52.829641Z","iopub.status.idle":"2022-03-27T07:32:52.834998Z","shell.execute_reply.started":"2022-03-27T07:32:52.829607Z","shell.execute_reply":"2022-03-27T07:32:52.834322Z"}}
# y

# %% [code] {"execution":{"iopub.status.busy":"2022-03-29T17:05:46.148862Z","iopub.execute_input":"2022-03-29T17:05:46.149627Z","iopub.status.idle":"2022-03-29T17:05:56.457317Z","shell.execute_reply.started":"2022-03-29T17:05:46.149585Z","shell.execute_reply":"2022-03-29T17:05:56.456541Z"}}
z = np.load("../input/imagearrray/X_51033.npz")
x = z['arr_0']
x /= 255

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.843781Z","iopub.execute_input":"2022-03-27T07:32:52.844328Z","iopub.status.idle":"2022-03-27T07:32:52.849495Z","shell.execute_reply.started":"2022-03-27T07:32:52.844274Z","shell.execute_reply":"2022-03-27T07:32:52.848788Z"}}
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.850869Z","iopub.execute_input":"2022-03-27T07:32:52.851493Z","iopub.status.idle":"2022-03-27T07:32:52.85666Z","shell.execute_reply.started":"2022-03-27T07:32:52.851453Z","shell.execute_reply":"2022-03-27T07:32:52.855868Z"}}
# x = np.array(x)

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.857873Z","iopub.execute_input":"2022-03-27T07:32:52.858378Z","iopub.status.idle":"2022-03-27T07:32:52.86413Z","shell.execute_reply.started":"2022-03-27T07:32:52.858342Z","shell.execute_reply":"2022-03-27T07:32:52.863446Z"}}
# gc.collect()

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.86577Z","iopub.execute_input":"2022-03-27T07:32:52.866268Z","iopub.status.idle":"2022-03-27T07:32:52.871248Z","shell.execute_reply.started":"2022-03-27T07:32:52.866231Z","shell.execute_reply":"2022-03-27T07:32:52.870488Z"}}
# x = np.array(x, np.float32)
# x

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.872221Z","iopub.execute_input":"2022-03-27T07:32:52.873373Z","iopub.status.idle":"2022-03-27T07:32:52.879842Z","shell.execute_reply.started":"2022-03-27T07:32:52.873335Z","shell.execute_reply":"2022-03-27T07:32:52.879138Z"}}
# x = preprocess_input(x)

# %% [code] {"execution":{"iopub.status.busy":"2022-03-29T18:34:04.858474Z","iopub.execute_input":"2022-03-29T18:34:04.859282Z","iopub.status.idle":"2022-03-29T18:34:04.866363Z","shell.execute_reply.started":"2022-03-29T18:34:04.859236Z","shell.execute_reply":"2022-03-29T18:34:04.865613Z"}}
 X.shape

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T07:32:52.892879Z","iopub.execute_input":"2022-03-27T07:32:52.893299Z","iopub.status.idle":"2022-03-27T07:32:52.89714Z","shell.execute_reply.started":"2022-03-27T07:32:52.893258Z","shell.execute_reply":"2022-03-27T07:32:52.89633Z"}}
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input


# %% [code] {"execution":{"iopub.status.busy":"2022-04-13T18:00:00.927382Z","iopub.execute_input":"2022-04-13T18:00:00.927645Z","iopub.status.idle":"2022-04-13T18:00:01.449756Z","shell.execute_reply.started":"2022-04-13T18:00:00.927618Z","shell.execute_reply":"2022-04-13T18:00:01.449106Z"}}
# Transfered learning with VGG16
base_model = VGG16(input_shape=(128, 128, 3), include_top=False)
flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(512, activation='relu')
d1 = Dropout(0.5)
predictions = Dense(y.shape[1], activation='softmax')
model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    d1,
    predictions
])
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()
gc.collect()

# %% [code] {"execution":{"iopub.status.busy":"2022-04-14T02:59:40.273513Z","iopub.execute_input":"2022-04-14T02:59:40.273791Z","iopub.status.idle":"2022-04-14T02:59:53.593416Z","shell.execute_reply.started":"2022-04-14T02:59:40.273749Z","shell.execute_reply":"2022-04-14T02:59:53.592779Z"}}
# Efficient net B7
from keras.applications.efficientnet import EfficientNetB7
base_model = EfficientNetB7(input_shape=(128, 128, 3), include_top=False)
flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(512, activation='relu')
d1 = Dropout(0.5)
predictions = Dense(904, activation='softmax')
model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    d1,
    predictions
])
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()
gc.collect()

# %% [code] {"execution":{"iopub.status.busy":"2022-04-14T05:19:51.182759Z","iopub.execute_input":"2022-04-14T05:19:51.183017Z","iopub.status.idle":"2022-04-14T05:19:56.333372Z","shell.execute_reply.started":"2022-04-14T05:19:51.182988Z","shell.execute_reply":"2022-04-14T05:19:56.332702Z"}}
import tensorflow as tf
base_model = tf.keras.applications.ResNet50(include_top=False,input_shape=(200, 200, 3))
flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(512, activation='relu')
d1 = Dropout(0.5)
predictions = Dense(196, activation='softmax')
model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    d1,
    predictions
])
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()
gc.collect()

# %% [code] {"execution":{"iopub.status.busy":"2022-04-14T02:44:35.454782Z","iopub.execute_input":"2022-04-14T02:44:35.455080Z","iopub.status.idle":"2022-04-14T02:44:42.922111Z","shell.execute_reply.started":"2022-04-14T02:44:35.455047Z","shell.execute_reply":"2022-04-14T02:44:42.920822Z"}}
model = Sequential()
model.add(Conv2D(32, (11, 11), activation='relu', name='C1', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2'))
model.add(Conv2D(16, (9, 9), activation='relu', name='C3'))
model.add(LocallyConnected2D(16, (9, 9), activation='relu', name='L4'))
model.add(LocallyConnected2D(16, (7, 7), strides=2, activation='relu', name='L5') )
model.add(LocallyConnected2D(16, (5, 5), activation='relu', name='L6'))
model.add(Flatten(name='F0'))
model.add(Dense(4096, activation='relu', name='F7'))
model.add(Dropout(rate=0.5, name='D0'))
model.add(Dense(2731, activation='softmax', name='F8'))
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()

# %% [code] {"papermill":{"duration":2.632205,"end_time":"2022-02-06T12:54:26.901997","exception":false,"start_time":"2022-02-06T12:54:24.269792","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-04-06T09:54:10.832531Z","iopub.execute_input":"2022-04-06T09:54:10.83279Z","iopub.status.idle":"2022-04-06T09:54:13.375052Z","shell.execute_reply.started":"2022-04-06T09:54:10.832758Z","shell.execute_reply":"2022-04-06T09:54:13.374366Z"}}
model = Sequential()

model.add(Conv2D(64, (6, 6), strides = (1, 1), input_shape = (128, 128, 3)))
model.add(BatchNormalization(axis = 3))
model.add(Activation('relu'))


model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), strides = (1,1)))
model.add(Activation('relu'))
model.add(AveragePooling2D((3, 3)))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(2731, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2022-04-13T18:00:16.296656Z","iopub.execute_input":"2022-04-13T18:00:16.296927Z"},"jupyter":{"outputs_hidden":true}}
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=40,use_multiprocessing=True,
)

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T09:28:30.36091Z","iopub.execute_input":"2022-03-27T09:28:30.36117Z","iopub.status.idle":"2022-03-27T09:28:30.367159Z","shell.execute_reply.started":"2022-03-27T09:28:30.361139Z","shell.execute_reply":"2022-03-27T09:28:30.366477Z"}}
# z = np.load("../output/kaggle/working/y.npz")

# %% [code] {"execution":{"iopub.status.busy":"2022-03-27T15:52:27.883311Z","iopub.execute_input":"2022-03-27T15:52:27.883921Z","iopub.status.idle":"2022-03-27T15:52:27.956859Z","shell.execute_reply.started":"2022-03-27T15:52:27.883886Z","shell.execute_reply":"2022-03-27T15:52:27.954355Z"}}
from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced'
                                               ,np.unique(y)
                                               ,y)

# %% [code] {"_kg_hide-output":true,"papermill":{"duration":880.068032,"end_time":"2022-02-06T13:09:06.987564","exception":false,"start_time":"2022-02-06T12:54:26.919532","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-04-14T05:22:03.294169Z","iopub.execute_input":"2022-04-14T05:22:03.294428Z","iopub.status.idle":"2022-04-14T05:32:31.402396Z","shell.execute_reply.started":"2022-04-14T05:22:03.294398Z","shell.execute_reply":"2022-04-14T05:32:31.401298Z"}}
history = model.fit(
    X,
    y,
    batch_size=64,
    epochs=40,
    verbose=1,

    validation_split=0.05,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    use_multiprocessing=True
)
# history = model.fit(X, y, epochs=100, batch_size=64, verbose=1)
model.save('./last.h5')

# fit(x, y, epochs=40, batch_size=64, verbose=1)

# %% [code]
gc.collect()

# %% [markdown] {"papermill":{"duration":4.297318,"end_time":"2022-02-06T13:09:23.881367","exception":false,"start_time":"2022-02-06T13:09:19.584049","status":"completed"},"tags":[]}
# ## Evaluation

# %% [code] {"execution":{"iopub.status.busy":"2022-04-14T05:48:18.713480Z","iopub.execute_input":"2022-04-14T05:48:18.713744Z","iopub.status.idle":"2022-04-14T05:48:18.927363Z","shell.execute_reply.started":"2022-04-14T05:48:18.713714Z","shell.execute_reply":"2022-04-14T05:48:18.926698Z"}}
plt.figure(figsize=(15,5))
train_error, = plt.plot(history.history['accuracy'], label='train_accuracy')
validation_error, = plt.plot(history.history['val_accuracy'], label='validation_accuracy')
plt.legend(handles=[train_error])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet50 on Killer Whale for 40 epochs. Execution time: 1000 s. Total Trainable parameters: 75,068,996")
plt.show() 

# %% [code] {"papermill":{"duration":4.830564,"end_time":"2022-02-06T13:09:32.769651","exception":false,"start_time":"2022-02-06T13:09:27.939087","status":"completed"},"tags":[]}
plt.figure(figsize=(15,5))
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# %% [code] {"papermill":{"duration":4.20966,"end_time":"2022-02-06T13:09:41.036508","exception":false,"start_time":"2022-02-06T13:09:36.826848","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-04-14T04:43:14.608655Z","iopub.execute_input":"2022-04-14T04:43:14.609326Z","iopub.status.idle":"2022-04-14T04:43:14.818127Z","shell.execute_reply.started":"2022-04-14T04:43:14.609289Z","shell.execute_reply":"2022-04-14T04:43:14.817471Z"}}
X = [0.0149, 0.0203, 0.0240, 0.0312, 0.0346, 0.0438, 0.0506, 0.0494, 0.0323, 0.0476, 0.0637, 0.0768, 0.0696, 0.1083, 0.1445, 0.1914, 0.2155, 0.2579, 0.3306, 0.3837, 0.4391, 0.4923, 0.5320, 0.5679, 0.5936, 0.6232, 0.6746, 0.6658, 0.6633, 0.6960 ]
y= [0.0037, 0.0111,0.0056,0.0037,0.0028,0.0093,0.0167,0.0148,0.0213,0.0287,0.0213,0.0130,0.0287,0.0343,0.0408,0.0361,0.0259,0.0324,0.0269,0.0269,0.0278,0.0306,0.0315,0.0278,0.0259,0.0306,0.0315,0.0269,0.0241,0.0232]
epoch = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
plt.figure(figsize=(15,5))
train_error, = plt.plot(X, label='train_accuracy')
validation_error, = plt.plot(y, label='validation_accuracy')
plt.legend(handles=[train_error])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet50 on dusky dolphin for 30 epochs. Execution time: 8400 s. Total Trainable parameters: 40,829,192")
plt.show() 

# %% [markdown] {"papermill":{"duration":4.032709,"end_time":"2022-02-06T13:09:49.421457","exception":false,"start_time":"2022-02-06T13:09:45.388748","status":"completed"},"tags":[]}
# ## inference

# %% [code] {"papermill":{"duration":4.57354,"end_time":"2022-02-06T13:09:58.270697","exception":false,"start_time":"2022-02-06T13:09:53.697157","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-03-27T09:28:30.803585Z","iopub.execute_input":"2022-03-27T09:28:30.804093Z","iopub.status.idle":"2022-03-27T09:28:30.807402Z","shell.execute_reply.started":"2022-03-27T09:28:30.804044Z","shell.execute_reply":"2022-03-27T09:28:30.806699Z"}}
# test = os.listdir("../input/happy-whale-and-dolphin/test_images")
# print(len(test))

# %% [code] {"papermill":{"duration":4.565778,"end_time":"2022-02-06T13:10:07.228743","exception":false,"start_time":"2022-02-06T13:10:02.662965","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-03-27T09:28:30.808761Z","iopub.execute_input":"2022-03-27T09:28:30.809237Z","iopub.status.idle":"2022-03-27T09:28:30.817921Z","shell.execute_reply.started":"2022-03-27T09:28:30.809197Z","shell.execute_reply":"2022-03-27T09:28:30.81715Z"}}
# col = ['image']
# test_df = pd.DataFrame(test, columns=col)
# test_df['predictions'] = ''
# #test_df=test_df.head(n=250)

# %% [code] {"papermill":{"duration":2490.403013,"end_time":"2022-02-06T13:51:42.266417","exception":false,"start_time":"2022-02-06T13:10:11.863404","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-03-27T09:28:30.820724Z","iopub.execute_input":"2022-03-27T09:28:30.820949Z","iopub.status.idle":"2022-03-27T09:28:30.82779Z","shell.execute_reply.started":"2022-03-27T09:28:30.820901Z","shell.execute_reply":"2022-03-27T09:28:30.826976Z"}}
# batch_size=5000
# batch_start = 0
# batch_end = batch_size
# L = len(test_df)

# while batch_start < L:
#     limit = min(batch_end, L)
#     test_df_batch = test_df.iloc[batch_start:limit]
#     print(type(test_df_batch))
#     X = Loading_Images(test_df_batch, test_df_batch.shape[0], "test_images")
#     X /= 255
#     predictions = model.predict(np.array(X), verbose=1)
#     for i, pred in enumerate(predictions):
#         p=pred.argsort()[-5:][::-1]
#         idx=-1
#         s=''
#         s1=''
#         s2=''
#         for x in p:
#             idx=idx+1
#             if pred[x]>0.7:
#                 s1 = s1 + ' ' +  label_encoder.inverse_transform(p)[idx]
#             else:
#                 s2 = s2 + ' ' + label_encoder.inverse_transform(p)[idx]
#         s= s1 + ' new_individual' + s2
#         s = s.strip(' ')
#         test_df.loc[ batch_start + i, 'predictions'] = s
#     batch_start += batch_size   
#     batch_end += batch_size
#     del X
#     del test_df_batch
#     del predictions
#     gc.collect()
    

# %% [code] {"papermill":{"duration":4.217125,"end_time":"2022-02-06T13:51:50.495542","exception":false,"start_time":"2022-02-06T13:51:46.278417","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-03-27T09:28:30.828937Z","iopub.execute_input":"2022-03-27T09:28:30.829336Z","iopub.status.idle":"2022-03-27T09:28:30.83718Z","shell.execute_reply.started":"2022-03-27T09:28:30.829298Z","shell.execute_reply":"2022-03-27T09:28:30.836276Z"}}
# test_df.to_csv('submission.csv',index=False)
# test_df.head()

# %% [code]


# %% [code]
