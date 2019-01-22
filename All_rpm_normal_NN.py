
# code for all rpms models with cross validation
#import numpy as np
import matplotlib.pyplot as plt
#import pandas
import math
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras as keras
import h5py
import pickle

## custom initializers: code below defines custom weight initializers for kernal and bias for keras function API used for MC version of training the model
#from keras import backend as K


def prediction_1(filename,input_1,output_1,x_test):
    model_1  = Model(inputs=inputs_1, outputs=output_1)
    model_1.load_weights(filename)
    #x1=model_1.get_weights()
    model_1.compile(optimizer=addm,loss='mean_squared_error',metrics=['accuracy'])
    ypred=model_1.predict(x_test)
    return ypred
## initializing constants and data extraction
#path_mat_file='C:\\Users\\dks0013\\Desktop\\Vibration exp-2017\\vibration-log-1\\july_endexp_finaldata\\sample_initial_data\\prelim results'
#path_mat_file='C:\\Users\\dks0013\\Desktop\\Vibration exp-2017\\vibration-log-1\\july_endexp_finaldata\\sample_initial_data\\try_all_rpm_data'
path_mat_file='C:\\Users\\dks0013\\Desktop\\Vibration exp-2017\\vibration-log-1\\july_endexp_finaldata\\sample_initial_data'
#file_name='\\Results_for_2400RPM_allgpm_improved_rpi4_prelim.mat'
file_name='\\cali_vali_data_all_rpm_var_rem_sel.mat'
#file_name='\\cali_vali_data_all_rpm_all_var_out_rem.mat'

import hdf5storage as hdf
rpm=np.arange(1800,1900,100)
j=0
ypred=[None]*len(rpm)
sypred=[None]*len(rpm)
vloss=[None]*len(rpm)
closs=[None]*len(rpm)
rmse_rpm=[None]*len(rpm)
rmse_flo=[None]*len(rpm)
rmse_flo1=[None]*len(rpm)
y_test=[None]*len(rpm)

for i in rpm:
    x_train=hdf.loadmat(path_mat_file+file_name)['x_{}c'.format(i)]
    x_test=hdf.loadmat(path_mat_file+file_name)['x_{}v'.format(i)]
    y_train=hdf.loadmat(path_mat_file+file_name)['y_{}c'.format(i)]
    y_test[j]=hdf.loadmat(path_mat_file+file_name)['y_{}v'.format(i)]
    #del path_mat_file, file_name
    #[ccc]=hdf.loadmat(path_mat_file+file_name,variable_names=['x1800_c','x1800_v','y1800_c','y1800_v'])

    perm=np.arange(0,len(x_train),1)
    random.seed(1000)
    random.shuffle(perm)
    nc=perm[0:int(np.floor(len(perm)*0.8))]
    nv=perm[(int(np.floor(len(perm)*0.8))+1):]
    x_cali=x_train[nc,:]
    x_vali=x_train[nv,:]
    y_cali=y_train[nc,:]
    y_vali=y_train[nv,:]
    
    batch_size=int(len(x_cali)/10)
    epochs=91
    freq_len=(len(x_cali[0])-0)
    
    # Preprocessing of data
    
    mncn_x=StandardScaler(with_std=True)     #only mean centering of data
    x_c_mn=mncn_x.fit_transform(x_cali)
    x_v_mn=mncn_x.transform(x_vali)
    #x_c_mn=x_c_mn1[perm,:]
    x_t_mn=mncn_x.transform(x_test)
    
    mncn_y=StandardScaler(with_std=True)     #only mean centering of data
    colsy=[1]
    y_c_mn=mncn_y.fit_transform(y_cali[:,colsy])
    y_v_mn=mncn_y.transform(y_vali[:,colsy])
    #y_c_mn=y_c_mn1[perm,:]
    #y_t_mn=mncn_y.transform(y_test)
    
    #Input layer
    
    inputs_1     = Input(shape=(freq_len,))

    lay_1       = Dense(10,activation='relu')(inputs_1)
    #lay_1       = Dense(5000,activation='relu')(inputs_1)
    #lay_1       = Dense(200)(inputs_1)
    
    lay_2       = Dense(1,activation='relu')(lay_1)
    #lay_2       = Dense(2000,activation='relu')(lay_1)
    
    lay_3       = Dense(521,activation='relu')(lay_2)
    #lay_3       = Dense(1000,activation='relu')(lay_2)
    #
    lay_4       = Dense(56,activation='relu')(lay_3)
    #lay_4       = Dense(500,activation='relu')(lay_3)
    
    lay_5       = Dense(1493,activation='relu')(lay_4)
    
    output_1     = Dense(units = 1)(lay_5)
    
    model_train  = Model(inputs=inputs_1, outputs=output_1)
    
    addm         =keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
    
    model_train.compile(optimizer=addm,loss='mean_squared_error',metrics=['accuracy'])
    model_train.summary()
    # checkpoint
    
    #sav_nam='model_bestval_{}rpm_models2.hdf5'.format(i)
    #filepath=sav_nam
    #checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #callbacks_list = [checkpoint]
    #model traning
    #history=model_train.fit(x_c_mn[:,:],y_c_mn[:,:], shuffle=True, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_data=(x_v_mn[:,:], y_v_mn[:,:]),verbose=1)
    history=model_train.fit(x_c_mn[:,:],y_c_mn[:,:], shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_v_mn[:,:], y_v_mn[:,:]),verbose=1)

#getting predictions
#model_1  = Model(inputs=inputs_1, outputs=output_1)
#model_1.load_weights(filepath)
#x1=model_1.get_weights()
#model_1.compile(optimizer=addm,loss='mean_squared_error',metrics=['accuracy'])
#pred=model_1.predict(x_t_mn[:,:])
##score=model_1.evaluate(x_t_mn[:,:],y_t_mn[:,:])
#rmse=math.sqrt(mean_squared_error(y_t_mn[:,:],pred))       #manual root mean squared error
#xx=np.zeros((epochs, 2))
#xx[:,1]=history.history['val_loss']      #traning history
#xx[:,0]=history.history['loss']
    ypred11=model_train.predict(x_t_mn[:,:])
    s1yp=mncn_y.inverse_transform(ypred11)
    rmse_flo1[j]=math.sqrt(mean_squared_error(y_test[j][:,1],s1yp[:,0]))
    
    ypred[j]=prediction_1(filepath,inputs_1,output_1,x_t_mn[:,:])
    sypred[j]=mncn_y.inverse_transform(ypred[j])
#    rmse_rpm[j]=math.sqrt(mean_squared_error(y_test[j][:,0],sypred[j][:,0]))       #manual root mean squared error
    rmse_flo[j]=math.sqrt(mean_squared_error(y_test[j][:,1],sypred[j][:,0]))
    #xx=np.zeros((epochs, 2))
    vloss[j]=history.history['val_loss']      #traning history
    closs[j]=history.history['loss']
    j=j+1
    

 #for running plot_model
#model_train.save('model_train_dense.h5')

#from keras.utils import plot_model
#plot_model(model_train,show_shapes=True,to_file='try.png')


## PLotting
##flowrate prediction comparison
#l=11;
#for k in np.arange(0,11):    
#    fl=plt.figure(l)
#    plt.scatter(np.arange(len(sypred[k])),sypred[k][:,0],label='Predicted')
#    plt.scatter(np.arange(len(sypred[k])),y_test[k][:,0],label='Measured')
#    #plt.xlabel('Sample Number',fontsize=15,fontweight='bold')
#    #plt.ylabel('Flowrate',fontsize=15,fontweight='bold')
#    #plt.title('Flowrate prediction',fontsize=17,fontweight='bold')
#    plt.xlabel('Sample Number',fontsize=15,fontweight='bold')
#    plt.ylabel('RPM',fontsize=15,fontweight='bold')
#    plt.title('RPM prediction',fontsize=17,fontweight='bold')
#    plt.legend(fontsize =17,prop=dict(weight='bold'))
#    mng = plt.get_current_fig_manager()
#    mng.window.showMaximized()
#    fl.show()
#    l=l+1
#    #RPM prediction 
#    r=plt.figure(l)
#    plt.scatter(np.arange(len(sypred[k])),sypred[k][:,0],label='Predicted')
#    plt.scatter(np.arange(len(sypred[k])),y_test[k][:,0],label='Measured')
#    plt.xlabel('Sample Number',fontsize=15,fontweight='bold')
#    plt.ylabel('RPM',fontsize=15,fontweight='bold')
#    plt.title('RPM prediction',fontsize=17,fontweight='bold')
#    plt.legend(fontsize =17,prop=dict(weight='bold'))
#    mng = plt.get_current_fig_manager()
#    mng.window.showMaximized()
#    r.show()
#    l=l+1
    #loss comparison
#    ls=plt.figure(k+2)
#    plt.plot(np.arange(epochs),closs[k][:,0],label='Cali_loss',linewidth=2)
#    plt.plot(np.arange(epochs),vloss[k][:,1],label='Vali_loss',linewidth=2)
#    plt.xlabel('Epochs',fontsize=15,fontweight='bold')
#    plt.ylabel('Loss (MSE)',fontsize=15,fontweight='bold')
#    plt.title('RPM prediction',fontsize=17,fontweight='bold')
#    plt.legend(fontsize =17,prop=dict(weight='bold'))
#    mng = plt.get_current_fig_manager()
#    mng.window.showMaximized()
#    ls.show()
    #l=l+1