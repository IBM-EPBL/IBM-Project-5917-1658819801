#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[251]:


get_ipython().system('pip install pandas')
import pandas as pd


# In[252]:


import numpy as np


# In[253]:


import matplotlib.pyplot as plt


# In[254]:


get_ipython().system('pip install tensorflow')
import tensorflow as tf


# In[255]:


get_ipython().system('pip install flask')


# In[256]:


get_ipython().system('pip install keras')
import keras as k


# In[257]:


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='DRFu3HjQ9ddubhLZ7sFDrA4hkr4KQHe8xDDaE0IkuuHk',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'crudeoilpriceprediction-donotdelete-pr-ag8hh8yru3ecwt'
object_key = 'Crude Oil Prices Daily.xlsx'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']

dataset= pd.read_excel(body.read())
dataset.head()


# In[258]:


dataset.head()


# In[259]:


dataset.isnull().any()


# In[260]:


dataset.isnull().sum()


# In[261]:


dataset.dropna(axis=0,inplace=True)


# In[262]:


dataset.isnull().sum()


# In[263]:


dataoil=dataset.reset_index()['Closing Value']
dataoil


# In[264]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
dataoil=scaler.fit_transform(np.array(dataoil).reshape(-1,1))
dataoil


# In[265]:


plt.plot(dataoil)


# In[266]:


training_size=int(len(dataoil)*0.65)
test_size=len(dataoil)-training_size
train_data,test_data=dataoil[0:training_size,:],dataoil[training_size:len(dataoil),:1]


# In[267]:


training_size,test_size


# In[268]:


train_data.shape


# In[269]:


def create_dataset(dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
        return np.array(dataX),np.array(dataY)


# In[270]:


time_step=10
x_train,y_train=create_dataset(train_data,time_step)
x_test,y_test=create_dataset(test_data,time_step)


# In[271]:


print(x_train.shape),print(y_train.shape)


# In[272]:


print(x_test.shape),print(y_test.shape)


# In[273]:


x_train


# In[274]:


x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)


# # MODEL BUILDING

# ## IMPORTING MODEL BUILDING LIBRARIES

# In[275]:


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error


# ## INITIALIZING THE MODEL

# In[276]:


model=Sequential()


# ## ADDING LSTM LAYERS 

# In[277]:


model.add(LSTM(50,return_sequences=True,input_shape=(10,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))


# ## adding output layers

# In[278]:


model.add(Dense(1))


# In[279]:


model.summary()


# ## configure the learning process 

# In[280]:


model.compile(loss='mean_squared_error',optimizer='adam')


# ## train the model

# In[281]:


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=50,batch_size=64,verbose=1)


# ## Model Evaluation

# In[282]:


regressor = Sequential()

regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units = 60))
regressor.add(Dropout(0.1))

regressor.add(Dense(units = 1))


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5)
history =regressor.fit(x_train, y_train, epochs = 20, batch_size = 15,validation_data=(x_test, y_test), callbacks=[reduce_lr],shuffle=False)


# In[283]:


train_predict = regressor.predict(x_train)
test_predict = regressor.predict(x_test)


# In[284]:


train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])


# In[285]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[286]:


from tensorflow.keras.models import load_model


# In[287]:


model.save("crude_oil.h5")


# In[288]:


get_ipython().system('tar -zcvf crude-oil-classification.tgz crude_oil.h5')


# In[289]:


look_back=10
trainPredictPlot=np.empty_like(dataoil)
trainPredictPlot[:,:]=np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,:]=train_predict
testPredictPlot=np.empty_like(dataoil)
testPredictPlot[:,:]=np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(dataoil)-1,:]=test_predict


# In[290]:


plt.plot(scaler.inverse_transform(dataoil))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[291]:


len(test_data)


# In[292]:


x_input=test_data[2866:].reshape(1,-1)
x_input.shape


# In[293]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[294]:


temp_input


# In[295]:


lst_output=[]
n_steps=10
i=0 
while(i<10):
    if(len(temp_input)>10):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input=x_input.reshape((1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input=x_input.reshape((1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1


# In[296]:


day_new=np.arange(1,11)
day_pred=np.arange(11,21)
len(dataoil)


# In[297]:


plt.plot(day_new,scaler.inverse_transform(dataoil[8206:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[298]:


df3=dataoil.tolist()
df3.extend(lst_output)
plt.plot(df3[8100:])


# In[299]:


df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)


# In[300]:


get_ipython().system('pip install ibm_watson_machine_learning')


# In[301]:


from ibm_watson_machine_learning import APIClient
wml_credentials = {
    "url" : "https://us-south.ml.cloud.ibm.com",
    "apikey":"utWAMW51Ru0mOOhhCxifxfB00ZM5BcE2MeoX_jBmqUlN"
}
client = APIClient(wml_credentials)


# In[302]:


wml_client = APIClient(wml_credentials)


# In[303]:


def guid_from_space_name(client, space_name):
    space = client.spaces.get_details()
    return(next(item for item in space['resources'] if item['entity']['name'] == space_name)['metadata']['id'])


# In[304]:


space_uid = guid_from_space_name(client, 'models')
print("Space UID =" + space_uid)


# In[305]:


client.set.default_space(space_uid)


# In[306]:


client.software_specifications.list()


# In[307]:


software_space_uid = client.software_specifications.get_uid_by_name("tensorflow_rt22.1-py3.9")
software_space_uid


# In[308]:


wml_client.spaces.list()


# In[309]:


SPACE_ID="66298be0-c720-4044-a231-796c15582cbe"


# In[310]:


wml_client.set.default_space(SPACE_ID)


# In[311]:


model_details = client.repository.store_model(model="crude-oil-classification.tgz", meta_props={
    client.repository.ModelMetaNames.NAME:"Crude Oil Prices Daily",
    client.repository.ModelMetaNames.TYPE:"tensorflow_2.7",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:software_space_uid
})
model_id = client.repository.get_model_uid(model_details)


# In[312]:


model_id


# In[313]:


x_train[0]


# In[314]:


regressor.predict([[0.11335703],
       [0.11661484],
       [0.12053902],
       [0.11550422],
       [0.1156523 ],
       [0.11683696],
       [0.1140234 ],
       [0.10980305],
       [0.1089886 ],
       [0.11054346]])


# In[319]:


client.repository.download(model_id,'crude_oil.h5.tar.gb')


# In[ ]:




