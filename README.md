# WeakRM
## weakly supervised learning of RNA modifications
Motivation: Increasing evidences suggest that post-transcriptional RNA modifications
regulate essential biomolecular functions and are related to the pathogenesis of various
diseases. Precise identification of RNA modification sites is essential for understanding
the regulatory mechanisms of RNAs. To date, many computational approaches have
been developed for the prediction of RNA modifications, most of which were based on
strong supervision. These approaches performed generally well on modifications with
base-resolution data, but behave problematic for modifications with only low-resolution
data, e.g., ac4C and hm5C.

Results: WeakRM is the first weakly supervised learning framework for predicting
RNA modifications from low-resolution epitranscriptome datasets, such as, those
generated from acRIP-seq and hMeRIP-seq. Evaluations on three independent datasets
(corresponding to three different RNA modification types and their sequencing technologies)
demonstrated the effectiveness of our approach in predicting RNA modifications from
low-resolution data. It outperformed state-of-the-art multi-instance learning methods for
genomic sequences, such as, WSCNN, which was originally designed for transcription
factor binding site prediction. Additionally, our approach captured motifs that are consistent
with existing knowledge, and visualization of the predicted modification-containing
regions unveiled the potentials of detecting RNA modifications with improved resolution.
## Requirements
- Python 3.x (3.8.8)
- Tensorflow 2.3.2
- Numpy 1.18.5
- scikit-learn 0.24.1
- Argparse 1.4.0
- prettytable 2.1.0  
  
``WeakRM`` was tested on the versions listed above, so we do not guarantee that it will work on different versions.
## Installation
Just clone this repository as follows.
```
git clone https://github.com/daiyun02211/WeakRM.git
cd ./WeakRM
```
## Usage
### Data pre-processing
First convert sequence stokens into bags using one-hot encoding
```
python ./Scripts/token2npy.py --input_dir='./Data/m7G/' --output_dir='./Data/m7G/processed/'
```
``token2npy`` reads the token data from ``--input_dir`` and outputs bag data to ``--output_dir``  
The instance length and stride can be adjusted by ``--len`` and ``--stride`` respectively, default values are 50 and 10.
### Training
```
python ./Scripts/main.py --training=True --input_dir='./Data/m7G/processed/'
```
where ``--input_dir`` is the directory where the processed data is stored  
Further parameters include:
- ``--epoch``: the number of epoch with default 20
- ``--lr_init``: the inital learning rate with default 1e-4
- ``--lr_decay``: the decayed learning rate with default 1e-5
- ``--saving``: whether save weights during training
- ``--cp_dir``: the path to checkpoint directory
### Evaluation
```
python ./Scripts/main.py --training=False --input_dir='./Data/m7G/processed/' 
```
when specifying ``--training`` as False, we can now evaluate the model performance  
the default checkpoints are stored in ``'./Data/m7G/processed/cp_dir/'``
## Illustration of the proposed framework
<p align="center">
  <img src="https://github.com/daiyun02211/WeakRM/blob/main/Img/net.jpg" width="50%" align="middle"/>
</p>  
