# Neural Network Matrix Factorization (NNMF)

* Tensorflow prototypes of Dziugaite and Roy's "Neural Network Matrix Factorization" (NNMF) model (https://arxiv.org/abs/1511.06443).
* I forked from [jstol's github repository](https://github.com/jstol/neural-net-matrix-factorization).
* I've tested on `Ubuntu 16.04` and `Python 3.5`.
  * This repository used `wget` and `unzip` module on ubuntu. Please confirm those module are already installed.

* model의 final layer에 sigmoid 를 추가하고 linear expansion을 통해, 1~5 사이의 결과를 얻을 수 있도록 매핑했다.
  * 논문에 final layer에 대한 설명이 없는 데, 임의로 추가했다. 하지만, 이거 추가하냐 안하냐에 따라서 결과가 별로 차이가 나질 않는다.
* ml-100k 데이터에 대해서 논문의 저자들은 0.907의 성능을 나타내고 있다고 말하고 있으나, 셋팅을 똑같이 해도 재현되지 않음.
  * `Matrix Factorization with Neural Networks and Stochastic Variational Inference` 논문에서 실험한 결과인 0.9380이 그나마 비슷한 성능인 것 같다.
  * 본인의 실험에서는 0.9426621927 의 결과를 나타내었음.
  * 실제 이 실험에서 내가 조절할 수 있는 부분은 lambda 값 밖에 없는 것 같다.
  * Final Layer는
    * 1. non-linear activation function 적용 안하기
    * 2. sigmoid 적용하기
    * 3. tanh 적용하기
  * 를 해봤는데, 2가 제일 좋은 것 같다.
  * activation function을 sigmoid에서 relu로 바꿔봤는데 성능이 하락함. 다른 parameter들이 물론 sigmoid에 맞춰져 있긴 했지만, 그래도 RMSE가 1.0 이상인 건 심하지 않았나.
  * activation이 sigmoid일 때 layer 갯수 늘리면 성능이 좋아지는 것 같긴 하다. 


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
