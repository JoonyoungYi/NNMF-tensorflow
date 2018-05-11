# Neural Network Matrix Factorization (NNMF)

## Overview
* Tensorflow prototypes of Dziugaite and Roy's "Neural Network Matrix Factorization" (NNMF) model (https://arxiv.org/abs/1511.06443).
* I forked from [jstol's github repository](https://github.com/jstol/neural-net-matrix-factorization).

## Performance

### ML 100K data

* In this paper, author said that RMSE was 0.907 in the ml-100k data.
* I reproduced this RMSE value. My result was 0.906. I used lambda 50.
* Jake Stolee claimed in `Matrix Factorization with Neural Networks and Stochastic Variational Inference` that that the RMSE performance of this paper is 0.9380 in the ml-100k data.
  * But they were wrong. I checked [this repository](https://github.com/jstol/neural-net-matrix-factorization) supposedly made by Jake Stolee and they did not use bias in the Fully connected layer and they did not use full batch and did not use RMSPropOptimizer.
  * I've changed these parts. This is why I can achieve author's performance on my experiments.


## Implementation Detail
* Authors did not explain the final layer of the model in the paper. I did not add activation function like sigmoid, relu to the final layer.
  * Because, Because not adding it has helped to improve performance.
* I've checked that clipping can enhance performance. Clipping means that clip the predicted rating to 5 when prediction is higher than 5, or to 1(or 0.5) when prediction is lower than 1.
  * I thought, in common sense, that it would be better to use clipping.
  * But, there was no significant difference between clipping and not doing so.
  * Rather, the non-clipping algorithm beats the clipping algorithm.
  * In my opinion, The algorithm presented in this paper is vulnerable to overfitting. Similar to other MLP-based papers, you can experience dramatic performance degradation by modifying the hyper-parameters presented in the paper.
  * Early stop plays a very important role in this implementation. In the case of ml-100k data, 2% validation set is used to prevent the model from overfitting. The clipped RMSE values affect the timing of early stop, so they are underfitting, so we think that using clipped RMSE is poor performance. (Actually, there was no big difference.)
  * If I use unclipped RMSE for early stop and use clipped RMSE to report performance, I might get better performance, but I did not test this method because it was an unfair comparison to other algorithms like PMF, LLORMA,. etc.


## Environment
* I've tested on `Ubuntu 16.04` and `Python 3.5`.
  * This repository used `wget` and `unzip` module on ubuntu. Please confirm those module are already installed.
* How to init:
```
virtualenv .venv -p python3
. .venv/bin/activate
pip install -r requirements.txt
```
* How to run:
```
. .venv/bin/activate
python run.py
```

## Etc
* If you any question, feel free to add an issue to issue tool in this repository. Thanks!
