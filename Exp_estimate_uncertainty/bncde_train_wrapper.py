import pickle
import torch
import pytorch_lightning as pl
from src.cancer_sim.cancer_simulation import get_cancer_sim_data
from src.cancer_sim.process_irregular_data import *
from src.cancer_sim.cancer_data_module import CancerDataloader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.models.bncde_model import BNCDE
from pytorch_lightning.loggers.csv_logs import CSVLogger


def bncde_train(hyperparams: dict,
                data_path,
                load_pickle,
                informative_sampling=False,
                device='cuda',
                accelerator='cuda',
                num_workers=0):

    torch.set_default_device("cuda")

    # 1) Instantiate ML flow logger
    csv_logger = CSVLogger(save_dir='./runs')
    csv_logger.log_hyperparams(hyperparams)

    # 2) Instantiate Bayesian Neural Controlled Differential Equation
    model = BNCDE(mc_samples=hyperparams['mc_samples'],hidden_size=hyperparams['hidden_size'],
                  sd_diffusion=hyperparams['sd_diffusion'], drift_layers=hyperparams['drift_layers'],
                  control_size=hyperparams['control_size'], treatment_size=hyperparams['treatment_size'],
                  prediction_window=hyperparams['prediction_window'],
                  interpolation_method=hyperparams['interpolation_method'],
                  learning_rate=hyperparams['learning_rate'], method=hyperparams['method'])

    # 3) Load or generate data
    with open(data_path, 'rb') as f:
        pickle_map = pickle.load(f)


    data = CancerDataloader(pickle_map=pickle_map, batch_size=hyperparams['batch_size'],
                            device=device, num_workers=num_workers)


    # 4) Instantiate callback and trainer
    earlystopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=hyperparams['patience'])

    trainer = pl.Trainer(accelerator=accelerator, min_epochs=1, max_epochs=hyperparams['max_epochs'],
                         gradient_clip_val=hyperparams['clip_grad'],
                         logger=csv_logger, devices=[0],
                         callbacks=[earlystopping_callback])

    # 5) Fit model
    trainer.fit(model, data)