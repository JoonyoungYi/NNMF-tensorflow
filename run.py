import random

from app.core import run

if __name__ == '__main__':
    for i in range(100):
        lambda_value = 0.02 * random.random()
        valid_rmse = run(lambda_value=lambda_value)

        with open('validation.txt', 'a') as f:
            f.write('{}\t{}\n'.format(lambda_value, valid_rmse))
