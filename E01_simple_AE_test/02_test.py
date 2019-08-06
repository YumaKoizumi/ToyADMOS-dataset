# -*- coding: utf-8 -*-
"""
 @file   02_test.py
 @brief  Test code of simple AE-based anomaly detection in sounds used experiment in [1].
 @author Yuma KOIZUMI (NTT Media Intelligence Labs., NTT Corp.)
 Coptright (C) 2019 NTT Media Intelligence Labs., NTT Corp. All right reserved.
 [1] Y. Koizumi, et al., “ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection,” in Proc. of WASPAA 2019.
"""
##################################################################################
import numpy as np  
import collections
from matplotlib import pylab as plt # for debug
#########################################################################
import chainer
from chainer import cuda, serializers
import os
import sys
import scipy.signal.windows
from tqdm import tqdm
import pandas as pd
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
DEVICE_NUM  = 2
xp          = cuda.cupy 
DEVICE_INFO = cuda.get_device_from_id( DEVICE_NUM )

###############################################################################
# load parameters
sp_param, dnn_param, training_param = Config.load_config()
# Dev set of normal sounds
toy_type = 'ToyCar'
toy_type = 'ToyConveyor'
toy_type = 'ToyTrain'
obs_dir  = './exp1_dataset_'+toy_type+'/train_normal/'
# save dir
dnn_dir  = './dnn_dir/'                  
# model file name
model_fn = toy_type+".h5"
# analysis condition
rho = 0.1 # FPR = 10%
# anomaly list
anomaly_cond_xlsx_dir = './anomaly_conditions/'
xlsx_fn               = anomaly_cond_xlsx_dir+toy_type+'_anomay_condition.xlsx'            
anm_cnd               = pd.read_excel( xlsx_fn )
# report file name
report_file = './'+toy_type+'_overlook_report.txt'     

################################################################################         
# Test wav files
dataset_dir  = './exp1_dataset_'+toy_type
nml_dir = dataset_dir+'/test_normal/'
anm_dir = dataset_dir+'/test_anomaly/'
# result dir
sav_dir = './results_'+model_fn+'/'
if(os.path.isdir(sav_dir)==False):
    os.mkdir(sav_dir)

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
    # return
    return score
    
###############################################################################
with cuda.Device( DEVICE_INFO ):
    # Window function
    win   = scipy.signal.windows.hann( sp_param["fftl"] )
    win   = cuda.to_gpu( win[:,np.newaxis] ).astype(np.float32)
    sp_param["win"] = win
    # Filter bank design
    BPF, cc = my_modules.melFilterBank(sp_param["fs"], sp_param["fftl"], dnn_param["NumFB"])
    BPF     = cuda.to_gpu(BPF.astype(np.float32))
    
###############################################################################
    # model definition
    dnn_model = model_definition.FCN_AE(
            dnn_param["NumFB"], 
            dnn_param["hid_dim"], 
            dnn_param["z_dim"], 
            dnn_param["num_hid"],
            dnn_param["num_fw"],
            dnn_param["num_bw"]
            ).to_gpu( DEVICE_INFO )
    serializers.load_hdf5(dnn_dir+model_fn, dnn_model)

    # Load testdata
    nml_all, fn_nml = my_modules.wav_read_test(nml_dir, sp_param["wav_read_gain"])
    anm_all, fn_anm = my_modules.wav_read_test(anm_dir, sp_param["wav_read_gain"])
###############################################################################
    X_set = my_modules.list_to_gpu( nml_all )
    A_set = my_modules.list_to_gpu( anm_all )
    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            MN = np.zeros( (len(X_set),) )
            MA = np.zeros( (len(A_set),) )
            print('Evaluating normal files...')
            for jj in tqdm( range( len(X_set) ) ):
                # normal
                x          = X_set[ jj ]
                score      = evaluate_wav( x ) 
                score      = cuda.to_cpu( score.data )
                svfn       = sav_dir+'nml_'+fn_nml[jj]+'.csv'
                np.savetxt(svfn, score, delimiter=",")
                MN[jj]     = np.max( score )
            print('Evaluating anomalous files...')
            for jj in tqdm( range( len(A_set) ) ):
                # anomaly
                x          = A_set[ jj ]
                score      = evaluate_wav( x ) 
                score      = cuda.to_cpu( score.data )
                svfn       = sav_dir+'anm_'+fn_anm[jj]+'.csv'
                np.savetxt(svfn, score, delimiter=",")
                MA[jj]     = np.max( score )
                
                
                
###############################################################################
# AUC evaluation
Thres = np.sort(MN)[::-1] 
TPR   = MA * 0
for jj in range( len(A_set) ):
    TPR[jj] = np.sum( MA > Thres[jj] ) / len(A_set)

plt.plot( np.linspace(0, 1, len(TPR)), TPR )
plt.plot( [rho, rho], [0, 1], 'r' )
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print( 'AUC: '+str( np.mean(TPR) ) )
###############################################################################
# F-measure
rho_index = int( len(Thres) * rho )
TP = TPR[rho_index] * len(TPR)
FP = int( rho * len(Thres) )
FN = len(TPR) - TP
Prec = TP / (TP + FP)
Recl = TP / (TP + FN)
Fmsr = (2 * Recl * Prec) / (Recl + Prec)
print('F-measure under FPR = '+str(rho*100)+'% conditon: '+str(Fmsr))
###############################################################################
# Overlooked files
threshold_rho = Thres[rho_index]
ovl_index     = np.where( MA < threshold_rho )[0]
ovl_list = []
for jj in range( len(ovl_index) ):
    ovl_fn = fn_anm[ovl_index[jj]].split('ab')
    ovl_list.append('ab'+ovl_fn[1][:2] )
c = collections.Counter( ovl_list )
# make report
with open(report_file, mode='w') as f:
    f.write('Overlooked files under FPR = '+str(rho*100)+'% conditon:\n')
    for kk in list(c.keys()):
        f.write('-------------------------------------\n')
        idx = int(kk[2:])-1
        for ii, cc in enumerate(anm_cnd.columns):
            f.write(cc+': '+anm_cnd.iloc[idx][ii]+'\n')
        f.write( 'Overlooked times: '+str(c[kk])+'\n' )
    f.write('-------------------------------------\n')
###############################################################################






