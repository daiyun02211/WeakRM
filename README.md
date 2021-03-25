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
## Illustration of the proposed framework
<p align="center">
  <img src="https://github.com/daiyun02211/WeakRM/blob/main/Img/net.jpg" width="50%" align="middle"/>
</p>  
