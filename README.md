# MEMTO (NeurIPS 2023)
MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection

Junho Song* · Keonwoo Kim* · Jeonglyul Oh · Sungzoon Cho (*Equal Contribution)

https://arxiv.org/abs/2312.02530

## Abstract
Detecting anomalies in real-world multivariate time series data is challenging due to complex temporal dependencies and inter-variable correlations. Recently, reconstruction-based deep models have been widely used to solve the problem. However, these methods still suffer from an over-generalization issue and fail to deliver consistently high performance. To address this issue, we propose the MEMTO, a memory-guided Transformer using a reconstruction-based approach. It is designed to incorporate a novel memory module that can learn the degree to which each memory item should be updated in response to the input data. To stabilize the training procedure, we use a two-phase training paradigm which involves using K-means clustering for initializing memory items. Additionally, we introduce a bi-dimensional deviation-based detection criterion that calculates anomaly scores considering both input space and latent space. We evaluate our proposed method on five real-world datasets from diverse domains, and it achieves an average anomaly detection F1-score of 95.74%, significantly outperforming the previous state-of-the-art methods. We also conduct extensive experiments to empirically validate the effectiveness of our proposed model's key components.


<p align="center">
<img src=".\png\MEMTO_figure.png" height = "350" alt="" align=center />
</p>
## Main Result
In the main experiment, we evaluate the performance of MEMTO on multivariate time series anomaly detection tasks by comparing it with 12 models.
**MEMTO achieves SOTA in multivariate time series anomaly detection tasks.**
<p align="center">
<img src=".\png\MEMTO_results.png" height = "450" alt="" align=center />
</p>

### Data
**LICENSE**: _by downloading our dataset you accept licenses of all its components. We
do not impose any new restrictions in addition to those licenses. You can find the list
of sources in the section "References" of our paper._
1. Download the data: `wget https://www.dropbox.com/s/o53umyg6mn3zhxy/data.tar.gz?dl=1 -O revisiting_models_data.tar.gz`
2. Move the archive to the root of the repository: `mv revisiting_models_data.tar.gz $PROJECT_DIR`
3. Go to the root of the repository: `cd $PROJECT_DIR`
4. Unpack the archive: `tar -xvf revisiting_models_data.tar.gz`
## data download
data download link = https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm

## Citation
If you find this repo useful, please cite our paper. 

```
@inproceedings{
anonymous2023memto,
title={{MEMTO}: Memory-guided Transformer for Multivariate Time Series Anomaly Detection},
author={Junho Song, Keonwoo Kim, Jeonglyul Oh, Sungzoon Cho},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=UFW67uduJd}
}
```
