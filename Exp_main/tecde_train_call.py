import yaml
from tecde_train_wrapper import tecde_train
from itertools import product
import numpy as np
np.random.seed(0)
# Load hyperparams
with open('src/config/tecde_hyperparam.yml', 'r') as stream:
    data = yaml.safe_load(stream)
keys, values = zip(*data.items())
values_tolist = [[p] if not isinstance(p, list) else p for p in values]
hyperparam_combinations = [dict(zip(keys, p)) for p in product(*values_tolist)]

# train ing
for i, hyperparams in enumerate(hyperparam_combinations):
    tecde_train(hyperparams=hyperparams,
                data_path='./data/cancer_sim/cancer_pickle_map_irregular.pkl',
                load_pickle=False,
                device='cuda',
                accelerator='cuda',
                num_workers=0)