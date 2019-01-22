
from hyperopt import fmin, tpe, hp, Trials
#import math
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import hdf5storage as hdf
## datasets
rpm=np.arange(1800,1900,100)
j=0
path_mat_file='C:\\Users\\dks0013\\Desktop\\Vibration exp-2017\\vibration-log-1\\july_endexp_finaldata\\sample_initial_data'
#file_name='\\Results_for_2400RPM_allgpm_improved_rpi4_prelim.mat'
file_name='\\cali_vali_data_all_rpm_var_rem_sel.mat'
for i in rpm:
    x_train=hdf.loadmat(path_mat_file+file_name)['x_{}c'.format(i)]
    x_test=hdf.loadmat(path_mat_file+file_name)['x_{}v'.format(i)]
    y_train=hdf.loadmat(path_mat_file+file_name)['y_{}c'.format(i)]
    y_test=hdf.loadmat(path_mat_file+file_name)['y_{}v'.format(i)]
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
    # Preprocessing of data
    del x_train, y_train
    mncn_x=StandardScaler(with_std=True)     #only mean centering of data
    x_c_mn=mncn_x.fit_transform(x_cali)
    x_v_mn=mncn_x.transform(x_vali)
    #x_c_mn=x_c_mn1[perm,:]
    x_t_mn=mncn_x.transform(x_test)
    
    mncn_y=StandardScaler(with_std=True)     #only mean centering of data
    colsy=[1]
    y_c_mn=mncn_y.fit_transform(y_cali[:,colsy])
    y_v_mn=mncn_y.transform(y_vali[:,colsy])
    
    del x_cali, x_vali, y_cali, y_vali


##search space defined for hyperopt
space={'epochs':(hp.quniform('epochs',20,100,1)),
       'layers':hp.choice('layers',[{'n_layers':1,
                                     'n_units_layer':[(hp.quniform('n_units_layer1',500,1500,1))]},
                                    {'n_layers':2,'n_units_layer':[(hp.quniform('n_units_layer21',10,1500,1)),
                                                                   (hp.quniform('n_units_layer22',10,1500,1))]},
                                    {'n_layers':3,'n_units_layer':[(hp.quniform('n_units_layer31',10,1500,1)),
                                                                   (hp.quniform('n_units_layer32',10,1500,1)),
                                                                   (hp.quniform('n_units_layer33',10,1500,1))]},
                                    {'n_layers':4,'n_units_layer':[(hp.quniform('n_units_layer41',10,1500,1)),
                                                                   (hp.quniform('n_units_layer42',10,1500,1)),
                                                                   (hp.quniform('n_units_layer43',10,1500,1)),
                                                                   (hp.quniform('n_units_layer44',10,1500,1))]},
                                    {'n_layers':5,'n_units_layer':[(hp.quniform('n_units_layer51',10,1500,1)),
                                                                   (hp.quniform('n_units_layer52',10,1500,1)),
                                                                   (hp.quniform('n_units_layer53',10,1500,1)),
                                                                   (hp.quniform('n_units_layer54',10,1500,1)),
                                                                   (hp.quniform('n_units_layer55',10,1500,1))]}])}


##Getting Deep neural network model of variable depth
def get_NN_model(input_len, layer_details):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Input
    
    n_lay=layer_details['n_layers']
    
    n_units=layer_details['n_units_layer']
    
    input1=Input(shape=(input_len,))
    lay=Dense(int(n_units[0]),activation='relu')(input1)    
    for layn in range(n_lay-1):
        lay=Dense(int(n_units[layn+1]),activation='relu')(lay)
        
    output1=Dense(units=1)(lay)
    model_train  = Model(inputs=input1, outputs=output1)    
    model_train.summary()
    return model_train
    

# training function to get loss value (function of hyperparameters)
def fn_nn(params):
    
    #from tensorflow.keras.callbacks import ModelCheckpoint
    import tensorflow.keras as keras    
    
    batch_size=int(len(x_c_mn)/10)
    epochs=int(params['epochs'])
    freq_len=(len(x_c_mn[0])-0)
    
    lay_details=params['layers']
    
    #model architecture
    #input1=Input(shape=(freq_len,))
    
    model_train=get_NN_model(freq_len,lay_details)
    
    addm=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
    
    model_train.compile(optimizer=addm,loss='mean_squared_error',metrics=['accuracy'])
    
    history=model_train.fit(x_c_mn[:,:],y_c_mn[:,:], shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_v_mn[:,:], y_v_mn[:,:]),verbose=1)
    
    vloss=history.history['val_loss'][-1] 
    
    return vloss

trials=Trials()
best=fmin(fn_nn,space,algo=tpe.suggest,max_evals=100, trials=trials)
print(best)
