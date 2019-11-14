from chainer.datasets import open_pickle_dataset


inp_path = '../../../data/processed.pkl'
with open_pickle_dataset(inp_path) as f:
    for i, data in enumerate(f):
        shape = data['elements'].shape
        if data['positions'].shape[:-1] != shape:
            print('positions')
            print(shape)
            print(data['symbols'].shape)
            print(data['positions'].shape)
        assert data['forces'].shape[:-1] == shape