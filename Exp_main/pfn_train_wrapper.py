import pickle
import torch
import pytorch_lightning as pl
from src.cancer_sim.cancer_simulation import get_cancer_sim_data
from src.cancer_sim.process_irregular_data import *
from src.cancer_sim.cancer_data_module import CancerDataloader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.models.TimePFN_GMM import TimePFN
from pytorch_lightning.loggers.csv_logs import CSVLogger


def pfn_train(hyperparams: dict,
			  data_path,
			  load_pickle,
			  informative_sampling=False,
			  device='cuda',
			  accelerator='cuda',
			  num_workers=0):
	
	torch.set_default_device('cuda')

	# 1) Instantiate logger
	csv_logger = CSVLogger(save_dir='./runs')
	csv_logger.log_hyperparams(hyperparams)

	# 2) Instantiate TimePFN with GMM Head
	model = TimePFN(seq_len=hyperparams['seq_len'], pred_len=hyperparams['pred_len'],
				 gmm_n_components=hyperparams['gmm_n_components'],
				 gmm_min_sigma=hyperparams['gmm_min_sigma'],
				 gmm_pi_temp=hyperparams['gmm_pi_temp'],
				 c_out=hyperparams['c_out'],
				 patch_size=hyperparams['patch_size'],
				 embed_dim=hyperparams['embed_dim'],
				 dropout=hyperparams['dropout'],
				 n_heads=hyperparams['n_heads'],
				 d_ff=hyperparams['d_ff'],
				 activation=hyperparams['activation'],
				 e_layers=hyperparams['e_layers'],
				 d_model=hyperparams['d_model'],
				 treatment_size=hyperparams['treatment_size'],
				 prediction_window=hyperparams['prediction_window'],
				 interpolation_method=hyperparams['interpolation_method'],
				 learning_rate=hyperparams['learning_rate'],
				 coverage_metric_confidence=hyperparams['coverage_metric_confidence'])
	
	# 3) Load or generate data
	with open(data_path, 'rb') as f:
		pickle_map = pickle.load(f)
	
	data = CancerDataloader(pickle_map=pickle_map, batch_size=hyperparams['batch_size'],
						 device=device, num_workers=num_workers)

	# 4) Instantiate callback and trainer
	earlystopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=hyperparams["patience"])

	trainer = pl.Trainer(accelerator=accelerator, min_epochs=1, max_epochs=hyperparams['max_epochs'],
					  gradient_clip_val=hyperparams['clip_grad'],
					  logger=csv_logger, devices=[0],
					  callbacks=[earlystopping_callback])
	
	# 5) Fit model
	trainer.fit(model, data)

