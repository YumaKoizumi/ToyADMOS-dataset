# -*- coding: utf-8 -*-
# Config ########################################################################
def load_config():
    # signal processing parameters
    sp_param = {
    "fs"    : 16000.0,
    "fftl"  : 512,
    "shift" : 256,
    "Log_reg" : 1e-8,
    "wav_read_gain" : 1.0
    }
    # DNN shape
    dnn_param = {
    "NumFB"   : 64,     # dim of mel-filter-bank
    "hid_dim" : 512,    # num of hidden units of FCN layer
    "z_dim"   : 128,    # dimension of compressed feature of encode
    "num_hid" : 4,      # num layers of FCN    
    "num_fw"  : 10,     # num of frame concat (future)
    "num_bw"  : 10,     # num of frame concat (past)
    }
    # training parameters
    training_param = {
    "set_size"          : 100,          # num of samples per one set
    "Backprop_per_file" : 2,            # num of samples per 1 backpropagation
    "MAX_EPOCH"         : 200,   
    "lr_base"           : 10**(-4),     # initial learning rate
    "lr_decal_start"    : 100,          # learning rate decrease epoch
    "lr_decay_factor"   : 100,          # final learning rate = 1/lr_decay_factor
    "l2_weight"         : 10**(-4),     # Weight decay
    }
    return sp_param, dnn_param, training_param

###############################################################################
if __name__ == "__main__":
     print('debug')   
    
    
    