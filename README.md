# Neural Network Matrix Factorization (NNMF)

* Tensorflow prototypes of Dziugaite and Roy's "Neural Network Matrix Factorization" (NNMF) model (https://arxiv.org/abs/1511.06443).
* I forked from [jstol's github repository](https://github.com/jstol/neural-net-matrix-factorization).
* How to init
```
mkdir data
cd data
wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
cd ..
virtualenv .venv -p python3
. .venv/bin/activate
pip install -r requirements.txt
```

* How to run
```
. .venv/bin/activate
python run.py
```
