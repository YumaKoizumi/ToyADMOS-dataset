# -*- coding: utf-8 -*-
"""
 @file   01_train.py
 @brief  Training code of simple AE-based anomaly detection in sounds used experiment in [1].
 @author Yuma KOIZUMI (NTT Media Intelligence Labs., NTT Corp.)
 Coptright (C) 2019 NTT Media Intelligence Labs., NTT Corp. All right reserved.
 [1] Y. Koizumi, et al., “ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection,” in Proc. of WASPAA 2019.
"""
import numpy as np  
#########################################################################
import chainer
from chainer import cuda, optimizers, serializers
import os
import sys
import scipy.signal.windows
from tqdm import tqdm
#########################################################################
sys.path.append( os.path.join(os.path.dirname(__file__), './Modules') )
import my_gpu_funcs as my_gpufncs             
import my_modules
import Config                                 # Parameters
import model_definition                       # DNN definitions

###################################
# Print environment
chainer.print_runtime_info()
# GPU setting
cuda.check_cuda_available()
DEVICE_NUM  = 0
xp          = cuda.cupy 
DEVICE_INFO = cuda.get_device_from_id( DEVICE_NUM )
debug_mode  = False 
if( debug_mode ):
    from matplotlib import pylab as plt 
    
###############################################################################
# load parameters
sp_param, dnn_param, training_param = Config.load_config()
# Dev set of normal sounds
toy_type = 'ToyConveyor'# 'ToyCar' or 'ToyConveyor' or 'ToyTrain'
obs_dir  = './exp1_dataset_'+toy_type+'/train_normal/'
# save dir
dnn_dir  = './dnn_dir/'                         
if(os.path.isdir(dnn_dir)==False):
    os.mkdir(dnn_dir)    

###############################################################################
# sub module
def evaluate_wav( x ):
    # FFT
    Xr, Xi = my_gpufncs.exe_fft(x, sp_param)
    ###############################################################
    # feature extraction
    fet   = my_gpufncs.feature_extraction(Xr, Xi, BPF, sp_param["Log_reg"]) 
    # Calculate score    
    score, x, y = dnn_model( fet )
    ###############################################################
    # Loss function
    loss = my_gpufncs.loss_MMSE(score) 
    ###############################################################
    # return
    return loss, score, x, y

def exe_one_set( X_set ):
    total_cnt  = 0
    sum_loss   = 0.0
    X_set_perm = np.random.permutation( len(X_set) )
    # Training start ########################################################## 
    bp_cnt  = 0
    xin    = xp.zeros((sp_param["fftl"],), dtype=np.float32)
    for jj in range( len(X_set) ):
        # random draw #########################################################
        sample_id = X_set_perm[jj]
        x         = X_set[sample_id]          
        xin       = my_gpufncs.concat_2_wavs(xin, x, sp_param["fftl"])
        bp_cnt    += 1                        
        if(bp_cnt == training_param["Backprop_per_file"]):
            # back propagation ################################################
            loss, score, x, y = evaluate_wav( xin[sp_param["fftl"]:] ) 
            dnn_model.cleargrads()        
            loss.backward()
            optm_dnn.update()
            loss.unchain_backward()
            sum_loss  += float(loss.data)
            total_cnt += 1
            bp_cnt     = 0
            xin        = xp.zeros((sp_param["fftl"],), dtype=np.float32) 
    return sum_loss/total_cnt, score, x, y

def debug_draw(score, x, y):
    plt.subplot(3,1,1)
    plt.imshow( np.flipud(cuda.to_cpu( x.data ).T), aspect='auto' )
    plt.subplot(3,1,2)
    plt.imshow( np.flipud(cuda.to_cpu( y.data ).T), aspect='auto' )
    plt.subplot(3,1,3)
    plt.plot( cuda.to_cpu( score.data ))
    plt.xlim([0, len(score)])
    plt.show()
    
###############################################################################
with cuda.Device( DEVICE_INFO ):    
    # Window function
    win   = scipy.signal.windows.hann( sp_param["fftl"] )
    win   = cuda.to_gpu( win[:,np.newaxis] ).astype(np.float32)
    sp_param["win"] = win
    # load data
    dev_set = my_modules.wav_read_trn(obs_dir, training_param["set_size"], sp_param["wav_read_gain"])
    # Filter bank design
    BPF, cc = my_modules.melFilterBank(sp_param["fs"], sp_param["fftl"], dnn_param["NumFB"])
    BPF     = cuda.to_gpu(BPF.astype(np.float32))

###############################################################################
print('Training start...')
with cuda.Device( DEVICE_INFO ):
    # model definition
    dnn_model = model_definition.FCN_AE(
            dnn_param["NumFB"], 
            dnn_param["hid_dim"], 
            dnn_param["z_dim"], 
            dnn_param["num_hid"],
            dnn_param["num_fw"],
            dnn_param["num_bw"]
            ).to_gpu( DEVICE_INFO )
    # Optimizer
    optm_dnn = optimizers.Adam(alpha=training_param["lr_base"], beta1=0.9, beta2=0.999, amsgrad=True)
    optm_dnn.setup( dnn_model )
    optm_dnn.add_hook(chainer.optimizer.WeightDecay( training_param["l2_weight"] ) ) 

    ###########################################################################
    # Start training
    for epoch in range(1, training_param["MAX_EPOCH"]+1):
        print("----------------------------------------------------------------")
        # Development
        set_perm  = np.random.permutation( len(dev_set) )
        for ii in tqdm( range( len(dev_set) ) ):
            X_set                 = my_modules.list_to_gpu( dev_set[ set_perm[ii] ] )
            sum_loss, score, x, y = exe_one_set( X_set )
        print("      epoch: "+str(epoch)+" - Development Loss = "+ str( sum_loss ))
        # lr decay
        if( epoch > training_param["lr_decal_start"] ):
            lr_subtract     = training_param["lr_base"] / training_param["lr_decay_factor"]
            lr_subtract    /= training_param["MAX_EPOCH"] - training_param["lr_decal_start"]
            optm_dnn.alpha -= lr_subtract
        # debug draw
        if( debug_mode ):
            debug_draw(score, x, y)
###############################################################################        
serializers.save_hdf5(dnn_dir+"/"+toy_type+".h5", dnn_model)        
               
    
    