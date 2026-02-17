# Run tests
import torch
import pytorch_lightning as pl
import os
from src.cancer_sim.cancer_data_module import CancerDataloader
from src.models.TimePFN_GMM import TimePFN
import pandas as pd
import pickle
import numpy as np


# helper function
def unpack_preds(pred_out):
    combined_data = {}
    losses = [batch['loss'] for batch in pred_out]
    y_preds = [batch['y_pred'] for batch in pred_out]
    if 'sigma_pred' in pred_out[0].keys():
        sigma_pred = [batch['sigma_pred'] for batch in pred_out]
    coverage = pred_out[-1]['coverage']
    size = pred_out[-1]['size']
    covariates = [batch['x'] for batch in pred_out]

    ys = [batch['y'] for batch in pred_out]
    combined_data['losses'] = torch.tensor(losses)
    combined_data['y_pred'] = torch.cat(y_preds, dim=0).squeeze(dim=-1)
    
    if 'sigma_preds' in pred_out[0].keys():
        combined_data['sigma_pred'] = torch.cat(sigma_pred, dim=0).squeeze(dim=-1)
    combined_data['y'] = torch.cat(ys, dim=0).squeeze()
    combined_data['coverage'] = coverage
    combined_data['size'] = size
    combined_data['covariates'] = torch.concatenate(covariates, dim=0)

    return combined_data




def eval_predictions(model_instance, model_path, data_path, batch_size=64,  informative_sampling=False,
                     device='cuda', num_workers=11, only_cf=False):

    # Load data
    with open(data_path, 'rb') as f:
        pickle_map = pickle.load(f)

    # Load state dict
    state_dict = torch.load(os.path.join(model_path))['state_dict']
    if informative_sampling:
        rm_keys = [key for key in state_dict.keys() if (key.startswith('z_field'))]
    else:
        rm_keys = [key for key in state_dict.keys() if (key.startswith('z_field') | key.startswith('obs_net'))]
    [state_dict.pop(key, None) for key in rm_keys]

    model_instance.load_state_dict(state_dict)
    trainer = pl.Trainer(accelerator=device, devices=[0])

    # Counterfactual data
    data_cf = CancerDataloader(pickle_map=pickle_map, batch_size=batch_size, device=device, num_workers=num_workers,
                               test_set='cf')
    pred_cf = trainer.predict(model_instance, data_cf)
    if only_cf:
        unpacked_cf = unpack_preds(pred_cf)
        y_cf = unpacked_cf['y']
        mu_cf = unpacked_cf['y_pred']
        return unpacked_cf['coverage'], unpacked_cf['size'], None, mu_cf, None, y_cf

    else:
        model_instance.coverage_metric.reset()

        # Factual data
        data_f = CancerDataloader(pickle_map=pickle_map, batch_size=batch_size,device=device, num_workers=num_workers,
                                  test_set='f')
        pred_f = trainer.predict(model_instance, data_f)


        # Unpack predictions
        unpacked_f = unpack_preds(pred_f)
        unpacked_cf = unpack_preds(pred_cf)

        # Get last treatment decision
        last_treatment_f = unpacked_f['covariates'][:, -1, -4:]
        last_treatment_cf = unpacked_cf['covariates'][:, -1, -4:]

        # Determine whether last treatment is in factual or counterfactual path (only relevant for augmented test data)
        treatment_1 = (last_treatment_f[:, 0] < last_treatment_cf[:, 0]) * 1  # 0 = CF, 1 = F

        # True CATE
        y_f = unpacked_f['y']
        y_cf = unpacked_cf['y']
        CATE = (y_f - y_cf) ** (-1) ** treatment_1
        observed = ~torch.isnan(CATE)
        CATE = CATE[observed]

        # Point estimates
        mu_f = unpacked_f['y_pred'][observed, :]
        mu_cf = unpacked_cf['y_pred'][observed, :]
        # For save
        y_f = y_f[observed]
        y_cf = y_cf[observed]

        return unpacked_f['coverage'], unpacked_f['size'], unpacked_cf['coverage'], unpacked_cf['size'], mu_f, mu_cf, y_f, y_cf


def run_tests(pfn_path,
              mc_samples=50,
              only_cf = False,
              window='one_step',
              informative_sampling=False,
              data_path='./data/cancer_sim/cancer_pickle_map_irregular.pkl',
              save=True,
              save_path='./data/results/',
              extension='',
              device='cuda',
              num_workers=11
              ):
    torch.set_default_device(device)

    results = {}
    batch_size = 5000 # does not impact prediction, only for computation speed
    window_dict = {'one_step':1, 'two_step':2, 'three_step':3, 'four_step':4, 'five_step':5}
    if informative_sampling:
        extension = extension+'_informative_sampling'

    ####################################################################################################################

    ####################################################################################################################
    # BNCDE

    # Iterate over multiple mc dropout configurations
    pfn_results = {}

    model_instance = TimePFN(seq_len=55, pred_len=window_dict[window], gmm_n_components=5, gmm_min_sigma=1e-3,
                             gmm_pi_temp=1.0, c_out=7, patch_size=2, embed_dim=64, dropout=0.1, 
                             n_heads=8, d_ff=512, activation='gelu', e_layers=2, d_model=128, treatment_size=4,
                             prediction_window=window_dict[window],
                             interpolation_method='linear', learning_rate=0.0001,
                             coverage_metric_confidence=0.95)
    instance_path = pfn_path
    if only_cf:
        bncde_coverage_f, bncde_size_f, bncde_coverage_cf, bncde_size_cf, mu_f, mu_cf, y_f, y_cf = eval_predictions(
            model_instance, instance_path, data_path, batch_size, only_cf=only_cf, informative_sampling=informative_sampling,device=device, num_workers=num_workers)
    else:
        bncde_coverage_f, bncde_size_f, bncde_coverage_cf, bncde_size_cf, mu_f, mu_cf, y_f, y_cf = eval_predictions(
            model_instance, instance_path, data_path, batch_size, informative_sampling=informative_sampling, device=device, num_workers=num_workers)
    pfn_results['None'] = { # 'None' = dropout_prob key level
        'coverage_f': bncde_coverage_f, 'size_f': bncde_size_f,
        'coverage_cf': bncde_coverage_cf, 'size_cf': bncde_size_cf,
        'mu_f': mu_f, 'mu_cf': mu_cf, 'y_f': y_f, 'y_cf': y_cf
    }

    # output structure
    results['pfn'] = pfn_results

    ####################################################################################################################

    # Extract predictions and outcomes into dictionary
    outcomes_dict = {}
    # Iterate through the dictionary and extract the errors
    for model, dropout_probs in results.items():
        outcomes = {}
        for dropout_prob, values in dropout_probs.items():
            outcomes[dropout_prob] = {}
            outcomes[dropout_prob]['mu_f'] = values['mu_f']
            outcomes[dropout_prob]['mu_cf'] = values['mu_cf']
            outcomes[dropout_prob]['y_f'] = values['y_f']
            outcomes[dropout_prob]['y_cf'] = values['y_cf']

        outcomes_dict[model] = outcomes

    # Exract coverage and size into pandas data frame
    # Create empty lists to store data
    model_list = []
    dropout_prob_list = []
    coverage_f_list = []
    size_f_list = []
    coverage_cf_list = []
    size_cf_list = []

    # Iterate through the dictionary and extract the values
    for model, dropout_probs in results.items():
        for dropout_prob, values in dropout_probs.items():
            model_list.append(model)
            dropout_prob_list.append(dropout_prob)
            coverage_f_list.append(values['coverage_f'])
            size_f_list.append(values['size_f'])
            coverage_cf_list.append(values['coverage_cf'])
            size_cf_list.append(values['size_cf'])

    # Create a pandas DataFrame
    coverage_df = pd.DataFrame({
                    'model': model_list,
                    'dropout_prob': dropout_prob_list,
                    'coverage_f': coverage_f_list,
                    'size_f': size_f_list,
                    'coverage_cf': coverage_cf_list,
                    'size_cf': size_cf_list
                })

    if save:

        coverage_df.to_csv(save_path+'/coverage_'+str(mc_samples)+'_'+window+extension+'.csv')

        torch.save(outcomes_dict, save_path + '/outcomes_' + str(mc_samples) + '_' + window +extension+'.pkl')

