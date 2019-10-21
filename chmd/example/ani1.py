import numpy as np
from chmd.models.ani import ANI1


def main():
    params = {
        'num_elements': 2,
        'aev_params':
        {
            'Rcr': 9.0,
            'Rca': 3.5,
            'EtaR': np.array([1.6e1]),
            'ShfR': np.array([9.0000000e-01, 1.1687500e+00, 1.4375000e+00,
                              1.7062500e+00, 1.9750000e+00, 2.2437500e+00,
                              2.5125000e+00, 2.7812500e+00, 3.0500000e+00,
                              3.3187500e+00, 3.5875000e+00, 3.8562500e+00,
                              4.1250000e+00, 4.3937500e+00, 4.6625000e+00,
                              4.9312500e+00]),
            'EtaA': np.array([8.0000000e+00]),
            'Zeta': np.array([3.2000000e+01]),
            'ShfA': np.array([9.0000000e-01, 1.5500000e+00, 2.2000000e+00,
                              2.8500000e+00]),
            'ShfZ': np.array([1.9634954e-01, 5.8904862e-01, 9.8174770e-01,
                              1.3744468e+00, 1.7671459e+00, 2.1598449e+00,
                              2.5525440e+00, 2.9452431e+00]),
        },
        'nn_params':
        {
            'n_layers': [[128, 128, 1], [128, 128, 1]],
        }

    }
    model = ANI1(**params)
    print(model)


if __name__ == "__main__":
    main()
