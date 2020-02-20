# ToyADMOS dataset
ToyADMOS dataset is a machine operating sounds dataset of approximately 540 hours of normal machine operating sounds and over 12,000 samples of anomalous sounds collected with four microphones at a 48kHz sampling rate, prepared by Yuma Koizumi and members in NTT Media Intelligence Laboratories. The ToyADMOS dataset is designed for anomaly detection in machine operating sounds (ADMOS) research. We have collected normal and anomalous operating sounds of miniature machines by deliberately damaging their components. It is designed for three tasks of ADMOS: product inspection (toy car), fault diagnosis for fixed machine (toy conveyor), and fault diagnosis for moving machine (toy train). For more information, refer to the paper [1]. If you use the ToyADMOS dataset in your work, please cite this paper where it was introduced.

>[1] Yuma Koizumi, Shoichiro Saito, Noboru Harada, Hisashi Uematsu and Keisuke Imoto, "ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection," in Proc of Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2019.
> Paper URL: https://arxiv.org/abs/1908.03299

## Download:
The dataset can be downloaded at https://zenodo.org/record/3351307#.XT-JZ-j7QdU. 

Since the total size of the ToyADMOS dataset is over 440GB, each sub-dataset is split into 7-9 files by 7-zip (7z-format). The total size of the compressed dataset is approximately 180GB, and that of each sub-dataset is approximately 60GB. Download the zip files corresponding to sub-datasets of interest and use your favorite compression tool to unzip these split zip files. 


## Detailed description of dataset
See the file named DETAIL.pdf

## Usage examples

To give a sense of the usage of this dataset, a set of Python codes for data-generation, training, and test are available. 

 <dl>
  <dt>Tutorials on small training/test datasets written in [1].</dt>
  <dd> - Dowload "C01_create_small_INT_dataset", "E01_simple_AE_test", and "anomaly_conditions"</dd>
  <dd> - Run "make_dataset_for_car_and_conveyor.py" and "make_dataset_for_train.py" in "C01_create_small_INT_dataset" to make dataset. 
<pre>
[20 Feb. 2020] Note that the description of the gain parameters in our paper was wrong.
Original: To control the signal-to-noise ratio, we multiplied 3.16 (+10 dB) by the waveforms of target sounds in the toy-car and toy-conveyor sub-datasets and by the waveforms of noise sounds in the toy-train sub-dataset. 
Correctl: To control the signal-to-noise ratio, we multiplied 3.16 (+10 dB) by the waveforms of target sounds in toy-train sub-dataset and by the waveforms of noise sounds in the toy-car and toy-conveyor sub-datasets.
</pre>
 </dd>
  <dd> - Run "01_train.py" in "E01_simple_AE_test" to train a model</dd>
  <dd> - Run "02_test.py" in "E01_simple_AE_test" to evaluate a model</dd>
  <dd> - Note that paths in each code need to be changed depending on your environment</dd>
</dl> 
  

We have tested these codes on follwoing environment:
<pre>
Python: 3.6.8
Chainer: 4.5.0
NumPy: 1.16.2
CuPy:
  CuPy Version          : 4.1.0
  CUDA Build Version    : 9000
  CUDA Driver Version   : 10000
  CUDA Runtime Version  : 9000
  cuDNN Build Version   : 7104
  cuDNN Version         : 7600
</pre>

## License: 
See the file named LICENSE.pdf

## Authors and Contact
- Yuma Koizumi ([Homepage](https://sites.google.com/site/yumakoizumiweb/profile-english), Email: <koizumi.yuma@ieee.org>)
- Shoichiro Saito
- Noboru Harada ([Homepage](http://www.kecl.ntt.co.jp/people/harada.noboru/index.html))
- Hisashi Uematsu
- Keisuke Imoto ([Homepage](https://sites.google.com/site/ksukeimoto/))
