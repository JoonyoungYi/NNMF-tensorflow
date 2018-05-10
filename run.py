import random

from app.core import run

if __name__ == '__main__':
    for i in range(10):
        lambda_value = 50
        batch_size = None
        valid_rmse, test_rmse = run(
            lambda_value=lambda_value, batch_size=batch_size)

        with open('test.txt', 'a') as f:
            f.write('{}\t{}\t{}\n'.format(lambda_value, valid_rmse, test_rmse))
