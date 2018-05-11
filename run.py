import random

from app.core import run

if __name__ == '__main__':

    # for i in range(1000):
    # lambda_value = random.random() * 100
    # Test only dropout
    # hyper_params = {
    #     'lambda_value': random.randint(1, 100),
    #     'hidden_layer_number': random.randint(3, 7),
    #     'K': random.randint(10, 100),
    #     'hidden_units_per_layer': random.randint(20, 100),
    #     'D_prime': random.randint(50, 100),
    #     'dropout_rate': random.random() * 0.5,
    # }
    hyper_params = {
        'lambda_value': 50,
        'hidden_unit_number': 50,
    }
    batch_size = None
    valid_rmse, test_rmse = run(batch_size=batch_size, **hyper_params)

    msg = '{}\t{}\t{}'.format('\t'.join(
        str(hyper_params[key])
        for key in sorted(hyper_params.keys())), valid_rmse, test_rmse)
    print(msg)
    with open('dropout.txt', 'a') as f:
        f.write(msg + '\n')
