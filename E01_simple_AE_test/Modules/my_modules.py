# -*- coding: utf-8 -*-
import numpy as np
import scipy as scipy
from scipy.io import wavfile
import glob 
from tqdm import tqdm
from chainer import cuda

cuda.check_cuda_available()
xp = cuda.cupy 

###################################################
def list_to_gpu(ll):
    rr = []
    for ii in range(len(ll)):
        rr.append( cuda.to_gpu( ll[ii] ) )
    return rr

def list_to_gpu_select_device(ll, DEVICE_INFO ):
    rr = []
    for ii in range(len(ll)):
        rr.append( cuda.to_gpu( ll[ii], device=DEVICE_INFO ) )
    return rr

###################################################
def wav_read_trn(obs_dir, wav_per_set, wav_read_gain):
    obs_files = glob.glob(obs_dir + "*.wav")
    Num_wav   = len( obs_files )
    TrnIndex  = np.random.permutation( Num_wav )
    obs_trn_all = []
    X_set = []
    cnt   = 0
    print( 'Loading... (Training set)' )
    for ii in tqdm( range( Num_wav ) ):
        x, org_fs = wavread( obs_files[TrnIndex[ii]] ) 
        X_set.append( x*wav_read_gain )
        cnt += 1
        if(cnt == wav_per_set):
            cnt = 0
            obs_trn_all.append( X_set )
            X_set = []
    return obs_trn_all

#################################################################################
def wav_read_test(wav_dir, wav_read_gain):
    wav_files = glob.glob(wav_dir + '*.wav')
    Num_wav   = len( wav_files )
    S_all  = []
    fn_all = []
    print( 'Loading... (Test set)' )
    for ii in tqdm( range( Num_wav ) ):
        fn = wav_files[ii]
        x, org_fs = wavread( fn ) 
        S_all.append( x*wav_read_gain )
        fn_all.append(fn[len(wav_dir):])
    return S_all, fn_all

# IO ########################################################################
def wavread(fn):
    fs, data = wavfile.read(fn)
    data     = (data.astype(np.float32) / 2**(15))
    return data, fs

def wavwrite(fn, data, fs):
    data = scipy.array(scipy.around(data * 2**(15)), dtype = "int16")
    wavfile.write(fn, fs, data)
    
# mel filter bank ###################################################################
def hz2mel(f):
    return 1127.01048 * np.log(f / 700.0 + 1.0)

def mel2hz(m):
    return 700.0 * (np.exp(m / 1127.01048) - 1.0)
    
def melFilterBank(fs, nfft, numChannels):
    fmax = fs / 2
    melmax = hz2mel(fmax)
    nmax = nfft / 2 + 1
    df = fs / nfft
    dmel = melmax / (numChannels + 1)
    melcenters = np.arange(1, numChannels + 1) * dmel
    fcenters = mel2hz(melcenters)
    indexcenter = fcenters // df
    if(indexcenter[0]==0):
        indexcenter[0] = 1
    for ii in range(1, len(indexcenter)):
        if(indexcenter[ii-1]>=indexcenter[ii]):
            indexcenter[ii] = indexcenter[ii-1] + 1
    indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
    indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))

    filterbank = np.zeros((int(numChannels), int(nmax)))
    for c in np.arange(0, numChannels):
        increment= 1.0 / (indexcenter[c] - indexstart[c])
        for i in np.arange(indexstart[c], indexcenter[c]):
            filterbank[c, int(i)] = (i - indexstart[c]) * increment
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in np.arange(indexcenter[c], indexstop[c]):
            filterbank[c, int(i)] = 1.0 - ((i - indexcenter[c]) * decrement)
            
    for c in np.arange(0, numChannels):
            filterbank[c, :] = filterbank[c, :] / (1e-8 + filterbank[c, :].sum())

    filterbank = filterbank.astype(np.float32)
    return filterbank, fcenters
    

##############################################################################
if __name__ == "__main__":
     print('debug')   
    
    
    