import random

from app.core import run

if __name__ == '__main__':

    for i in range(1000):
        # lambda_value = random.random() * 100
        # Test only dropout
        hyper_params = {
            # 'lambda_value': random.random() * 100,
            'lambda_value': 39.8,
            # 'hidden_layer_number': random.randint(3, 7),
            'hidden_layer_number': 4,
            # 'K': random.randint(10, 100),
            'K': 1,
            # 'D_prime': random.randint(50, 100),
            'D_prime': 60,
            'hidden_unit_number': 50,
            # 'dropout_rate': random.random() * 0.5,
            'learning_rate': 1e-3,
        }
        # hyper_params = {
        #     'lambda_value': 50,
        #     'hidden_unit_number': 50,
        # }
        batch_size = None
        valid_rmse, test_rmse = run(batch_size=batch_size, **hyper_params)

        print('\t'.join(sorted(hyper_params.keys())))
        msg = '{}\t{}\t{}'.format('\t'.join(
            str(hyper_params[key])
            for key in sorted(hyper_params.keys())), valid_rmse, test_rmse)
        print(msg)
        with open('results/ml-100k-4hl.txt', 'a') as f:
            f.write(msg + '\n')
