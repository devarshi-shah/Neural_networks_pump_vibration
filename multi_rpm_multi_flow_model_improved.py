
# code for all rpms models with cross validation
#import numpy as np
import matplotlib.pyplot as plt
#import pandas
import math
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
#import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras as keras
import h5py
import pickle
from h5py_datageneration_class import DataGenerator


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
#file_name='\\cali_vali_data_all_rpm_var_rem_sel.mat'
#file_name='\\cali_vali_data_all_rpm_all_var_out_rem.mat'

#import hdf5storage as hdf
rpm=np.arange(1500,2600,100)


def mean_std_part(filename,ch_size):
    f1=h5py.File(filename,'r')
    xsh=f1['x_cali'].shape
    ysh=f1['y_cali'].shape
    xvsh=f1['x_vali'].shape
    yvsh=f1['y_vali'].shape
    
    
    ch_size=200     #number of samples of data to be loaded at one time
    sx=np.zeros((ch_size,xsh[1]))       #initialize sum of x
    sxsq=np.zeros((ch_size,xsh[1]))
    sy=np.zeros((ch_size,2)) 
    sysq=np.zeros((ch_size,2))     #initialize sum of x^2 used later to find std.
    
    temp_len=np.arange(0,xsh[0],ch_size)
    temp_len=np.append(temp_len,xsh[0])
    
    for i in range(len(temp_len)-1):
        xtemp=f1['x_cali'][temp_len[i]:temp_len[i+1],:]
        ytemp=f1['y_cali'][temp_len[i]:temp_len[i+1],0:2]
        sht=xtemp.shape
        if sht[0]<(ch_size-1):
            zzx=np.zeros((ch_size-sht[0],xsh[1]))
            zzy=np.zeros((ch_size-sht[0],2))
            xtemp=np.concatenate((xtemp,zzx))
            ytemp=np.concatenate((ytemp,zzy))
            
        sx=np.add(sx,xtemp)
        ex_sq=np.square(xtemp)
        sxsq=np.add(sxsq,ex_sq)
        
        sy=np.add(sy,ytemp)
        ey_sq=np.square(ytemp)
        sysq=np.add(sysq,ey_sq)
    
    f1.close()  
    #mean calculation    
    mnx=np.sum(sx,axis=0)/(xsh[0]-1)        #mean of x
    mny=np.sum(sy,axis=0)/(xsh[0]-1)        #mean of y
    
    ##Std calculation
    stdx=np.sqrt(np.subtract((np.sum(sxsq,axis=0)/(xsh[0]-1)),np.square(mnx)))
    stdy=np.sqrt(np.subtract((np.sum(sysq,axis=0)/(xsh[0]-1)),np.square(mny)))  
    return mnx, mny, stdx, stdy, xsh, ysh, xvsh, yvsh


filename='multi_model_data_cali_vali_test.h5'
mnx,mny,stdx,stdy,xsh, ysh,xvsh,yvsh=mean_std_part(filename,150)
batch_size=int(xsh[0]/50)
epochs=100
freq_len=(xsh[1])
f1=h5py.File(filename,'r')

training_gen=DataGenerator(f1,xsh,ysh,data_type='cali',batch_size=batch_size,meanx=mnx,meany=mny,stdx=stdx,stdy=stdy)
validation_gen=DataGenerator(f1,xvsh,yvsh,data_type='vali',batch_size=batch_size,meanx=mnx,meany=mny,stdx=stdx,stdy=stdy)
#f1.close()
#Input layer

inputs_1     = Input(shape=(freq_len,))

#lay_1       = Dense(1000,activation='relu')(inputs_1)
lay_1       = Dense(7000,activation='relu')(inputs_1)
#lay_1       = Dense(200)(inputs_1)

#lay_2       = Dense(800,activation='relu')(lay_1)
lay_2       = Dense(5000,activation='relu')(lay_1)

#lay_3       = Dense(500,activation='relu')(lay_2)
lay_3       = Dense(2000,activation='relu')(lay_2)
#
#lay_4       = Dense(200,activation='relu')(lay_3)
lay_4       = Dense(800,activation='relu')(lay_3)

lay_5       = Dense(400,activation='relu')(lay_4)

lay_6       = Dense(200,activation='relu')(lay_5)

output_1     = Dense(units = 2)(lay_6)

model_train  = Model(inputs=inputs_1, outputs=output_1)

addm         =keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)

model_train.compile(optimizer=addm,loss='mean_squared_error',metrics=['accuracy'])
model_train.summary()
# checkpoint

sav_nam='model_bestval_all_rpm_models_l5_trial.hdf5'
filepath=sav_nam
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
#model traning
history=model_train.fit_generator(generator=training_gen, shuffle=True, epochs=epochs, callbacks=callbacks_list, validation_data=validation_gen,verbose=1,max_queue_size=3)
f1.close()
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
#ypred=prediction_1(filepath,inputs_1,output_1,x_t_mn[:,:])
#sypred=mncn_y.inverse_transform(ypred)
#rmse_rpm=math.sqrt(mean_squared_error(y_testf[:,0],sypred[:,0]))       #manual root mean squared error
#rmse_flo=math.sqrt(mean_squared_error(y_testf[:,1],sypred[:,1]))
##xx=np.zeros((epochs, 2))
#vloss=history.history['val_loss']      #traning history
#closs=history.history['loss']
#    
#f=open('multi_model_all_freq_lay5_res.pkl','wb')
#pickle.dump({'closs':closs,'ypred':ypred,'sypred':sypred,'epochs':epochs,'rmse_flo':rmse_flo,'rmse_rpm':rmse_rpm,'rpm':rpm,'vloss':vloss,'y_test':y_testf},f)
#f.close()
# #for running plot_model
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