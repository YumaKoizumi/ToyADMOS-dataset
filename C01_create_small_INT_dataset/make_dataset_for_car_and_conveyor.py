# -*- coding: utf-8 -*-
"""
 @file   make_dataset_for_car_and_conveyor.py
 @brief  Create a training/test smal dataset used in [1].
 @author Yuma KOIZUMI (NTT Media Intelligence Labs., NTT Corp.)
 Coptright (C) 2019 NTT Media Intelligence Labs., NTT Corp. All right reserved.
 [1] Y. Koizumi, et al., “ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection,” in Proc. of WASPAA 2019.
"""
###############################################################################
import os
import glob 
#
import numpy as np
import scipy as scipy
from scipy.io import wavfile
import librosa 
from tqdm import tqdm
###############################################################################
# Fix seed
np.random.seed(0)

# Parameters
dataset_base      = "../../ToyADMOS" # Please set ToyADMOS directory in your environment
subdataset        = "ToyConveyor"    # "ToyCar" or "ToyConveyor"
case_num          = "case1"
ch_num            = "ch1"
target_fs         = 16000
num_train_samples = 1000
target_gain       = 10**( 0/20) # 0 dB for ToyCar, & ToyConveyor
noise_gain        = 10**(10/20) # + 10 dB for ToyCar & Conveyor
normal_dir        = dataset_base+"/"+subdataset+"/"+case_num+"/NormalSound_IND/*"+ch_num
anomaly_dir       = dataset_base+"/"+subdataset+"/"+case_num+"/AnomalousSound_IND/*"+ch_num
if( subdataset == "ToyCar" ):
    noise_dir         = dataset_base+"/"+subdataset+"/EnvironmentalNoise_CNT/*"+ch_num
else:
    noise_dir         = dataset_base+"/"+subdataset+"/EnvironmentalNoise_CNT/*"+case_num+"*"+ch_num

# Save dir
save_dir        = "./exp1_dataset_"+subdataset 
trn_normal_dir  = save_dir+"/train_normal/"
tst_normal_dir  = save_dir+"/test_normal/"
tst_anomaly_dir = save_dir+"/test_anomaly/"
      
###############################################################################
# sub modules
def make_dir( dir_name ):  
    if(os.path.isdir(dir_name)==False):
        print("Make directory: "+dir_name)
        os.mkdir(dir_name)    
    
def wavread(fn):
    fs, data = wavfile.read(fn)
    data     = (data.astype(np.float32) / 2**(15))
    return data, fs

def wavwrite(fn, data, fs):
    data = scipy.array(scipy.around(data * 2**(15)), dtype = "int16")
    wavfile.write(fn, fs, data)

def wav_read_all(wav_dir, target_fs):
    wav_files = glob.glob(wav_dir + '*.wav')
    Num_wav   = len( wav_files )
    S_all  = []
    fn_all = []
    print( 'Loading...' )
    for ii in tqdm( range( Num_wav ) ):
        fn             = wav_files[ii]
        signal, org_fs = wavread( fn ) 
        if(org_fs != target_fs):
            signal = librosa.core.resample( signal, org_fs, target_fs )
        S_all.append(signal)
        fn_all.append(fn[len(wav_dir):])
    return S_all, fn_all

def load_and_cut_noise(N_all, ls):
    n_id = np.random.randint( len(N_all) )
    n    = N_all[n_id]
    if(len(n) > len(s)):
        ln = len(n)
        st = int( (ln-ls-1)*np.random.rand(1) )
        n  = n[st:st+ls]
    return n
    
###############################################################################
# make save dir
make_dir(save_dir)    
make_dir(trn_normal_dir)    
make_dir(tst_normal_dir)    
make_dir(tst_anomaly_dir)    

# load dataset
print("Normal samples")
S_all, sfn_all = wav_read_all(normal_dir, target_fs)
print("Anomalous samples")
A_all, afn_all = wav_read_all(anomaly_dir, target_fs)
print("Noise samples")
N_all, nfn_all = wav_read_all(noise_dir, target_fs)

###############################################################################
print("Save normal samples")
# Shuffle normal data
normal_perm = np.random.permutation( len( S_all ) )
for ii in range( len( S_all ) ):
    # load normal sample
    s    = S_all[ normal_perm[ii] ]
    sfn  = sfn_all[ normal_perm[ii] ]
    ls   = len(s)
    # load noise sample
    n    = load_and_cut_noise(N_all, ls)
    # mix samples
    x = target_gain * s + noise_gain * n
    # save samples
    if( ii < num_train_samples ):
        wavwrite(trn_normal_dir+sfn, x, target_fs)
    else:
        wavwrite(tst_normal_dir+sfn, x, target_fs)
    
###############################################################################
print("Save anomalous samples")
for ii in range( len( A_all ) ):
    # load normal sample
    s    = A_all[ ii ]
    sfn  = afn_all[ ii ]
    ls   = len(s)
    # load noise sample
    n    = load_and_cut_noise(N_all, ls)
    # mix samples
    x = target_gain * s + noise_gain * n
    # save samples
    wavwrite(tst_anomaly_dir+sfn, x, target_fs)    




