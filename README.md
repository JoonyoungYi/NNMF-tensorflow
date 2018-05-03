# Neural Network Matrix Factorization (NNMF)

* Tensorflow prototypes of Dziugaite and Roy's "Neural Network Matrix Factorization" (NNMF) model (https://arxiv.org/abs/1511.06443).
* I forked from [jstol's github repository](https://github.com/jstol/neural-net-matrix-factorization).
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
