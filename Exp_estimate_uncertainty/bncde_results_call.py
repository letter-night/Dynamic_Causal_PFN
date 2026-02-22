# Call results
import numpy as np
import yaml
from bncde_results_main import run_tests

with open('src/config/bncde_results.yml', 'r') as stream:
    config = yaml.safe_load(stream)


for i in [1,2,3,4,5]:
    np.random.seed(i)
    run_tests(bncde_path=config['bncde_path'],
              informative_sampling=config['informative_sampling'],
              window=config['window'],
              data_path='./data/cancer_sim/cancer_pickle_map_irregular.pkl',
              mc_samples=100,
              save=True, save_path='data/results/',
              extension='_seed'+str(i),
              device='cuda', num_workers=11)