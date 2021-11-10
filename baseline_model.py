#!/usr/bin/env python
# coding: utf-8

# In[23]:


#gpu setting

#pip install tensorflow-gpu
#pip install plaidml-keras plaidbench


# In[2]:


#GPU 사용 가능 여부 확인
from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[5]:


tf.test.is_gpu_available()


# In[6]:


#gpu 선택
#plaidml-setup


# In[ ]:





# In[ ]:





# In[4]:


import numpy as np
import pandas as pd

import zipfile
import glob
import torch
import matplotlib.pylab as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, concatenate, Input, ConvLSTM2D
from tensorflow.keras import Model

import warnings
warnings.filterwarnings("ignore")


# In[8]:


#conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch


# In[28]:


#data load

#압축 풀기
#train_files = zipfile.ZipFile('../radar_image_predict/data/train.zip')
#train_files.extractall('.')
#test_files = zipfile.ZipFile('../radar_image_predict/data/test.zip')
#test_files.extractall('../radar_image_predict/data/test')


# In[6]:


#data_load
train_files = glob.glob('../radar_image_predict/data/train/*.npy')
len(train_files)


# In[29]:


test_files = sorted(glob.glob('../radar_image_predict/data/test/*.npy'))
len(test_files)


# In[7]:


#data check
data_1st = np.load(train_files[0])
data_1st.shape


# In[ ]:





# In[8]:


#data visualization
#colormap setting
color_map = plt.cm.get_cmap('RdBu')
color_map = color_map.reversed()

#visualization
plt.style.use('fivethirtyeight')
plt.figure(figsize = (20,20))

for i in range(4):
    plt.subplot(1,5,i+1)
    plt.imshow(data_1st[:,:,i],cmap = color_map)

plt.subplot(1,5,5)
plt.imshow(data_1st[:,:,-1], cmap = color_map)
plt.show()


# In[ ]:





# In[17]:


#data preprocessing
def trainGenerator():
    for file in train_files:
        dataset = np.load(file)
        target= dataset[:,:,-1].reshape(120,120,1)
        remove_minus = np.where(target < 0, 0, target)
        feature = dataset[:,:,:4]

        yield (feature, remove_minus)
        
        
train_data = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32), (tf.TensorShape([120,120,4]),tf.TensorShape([120,120,1])))
train_dataset = train_data.batch(4).prefetch(1)
train_d = train_data


# In[ ]:





# In[18]:


#gpu 확인

print(f'tf.__version__: {tf.__version__}')

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)


# In[19]:


#modeling
def base_model(input_layer, start_neurons):
    
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    pool1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    pool2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(pool2)

    convm = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    output_layer = Conv2D(1, (1,1), padding="same", activation='relu')(uconv1)
    
    return output_layer

input_layer = Input((120, 120, 4))
output_layer = base_model(input_layer,64)


# In[20]:


device_name = tf.test.gpu_device_name()
print("Found GPU at:{}".format(device_name))


# In[21]:


model = Model(input_layer, output_layer)
model.compile(loss='mae', optimizer='adam')


# In[22]:


#gpu 사용하여 학습해보자
with tf.device("/device:GPU:0"):
    model.fit(train_dataset, epochs = 1, verbose=1)


# In[ ]:





# In[25]:





# In[30]:


#test데이터 만들기
X_test = []

for file in tqdm(test_files, desc = 'test'):
    data = np.load(file)
    X_test.append(data)

X_test = np.array(X_test)


# In[31]:


X_test.shape


# In[ ]:





# In[32]:


#모델 예측
pred = model.predict(X_test)


# In[ ]:





# In[34]:


#제출본 만들기
submission = pd.read_csv('../radar_image_predict/data/sample_submission.csv')


# In[35]:


submission.iloc[:,1:] = pred.reshape(-1, 14400).astype(int)
submission.to_csv('../radar_image_predict/data/Dacon_baseline.csv', index = False)


# In[ ]:




