import argparse
import logging
import os

import wandb

from src.utils.cancer_simulation import get_cancer_sim_data
from src.utils.data_utils import process_data, read_from_file, write_to_file
from src.utils.process_irregular_data import *
from TimePFN_trainer import trainer 

os.environ["WANDB_API_KEY"] = "YOUR WANDB API KEY"
wandb_entity = "YOUR WANDB ENTITY"

def init_arg():
	parser = argparse.ArgumentParser(description="TimePFN")
	parser.add_argument("--chemo_coeff", default=8, type=int)
	parser.add_argument("--radio_coeff", default=8, type=int)
	parser.add_argument("--experiment", type=str, default="TimePFN_default")
	parser.add_argument("--results_dir", default="results")
	parser.add_argument("--model_name", default="TmePFN")
	parser.add_argument("--multistep", default=False)
	parser.add_argument("--kappa", type=int, default=1)
	parser.add_argument("--lambda_val", type=float, default=1)
	parser.add_argument("--max_samples", type=int, default=1)
	parser.add_argument("--max_horizon", type=int, default=5)	

	parser.add_argument("--seq_len", type=int, default=58, help='input sequence length')
	parser.add_argument("--pred_len", type=int, default=59, help='prediction sequence length')
	
	parser.add_argument("--enc_in", type=int, default=6, help='encoder input size')
	parser.add_argument("--dec_in", type=int, default=6, help='decoder input size')
	parser.add_argument("--c_out", type=int, default=6, help='output size')
	parser.add_argument("--d_model", type=int, default=128, help='dimension of model')
	parser.add_argument("--n_heads", type=int, default=8, help='num of heads')
	parser.add_argument("--e_layers", type=int, default=2, help='num of encoder layers')
	parser.add_argument("--d_ff", type=int, default=512, help='dimension of fcn')
	parser.add_argument("--dropout", type=float, default=0.1, help='dropout')
	parser.add_argument("--activation", type=str, default='gelu', help='activation')
	parser.add_argument('--output_attention', default=False, help='whether to output attention in encoder')
	parser.add_argument("--patch_size", type=int, default=2, help='patch size')
	parser.add_argument("--embed_dim", type=int, default=64, help='embedding dimension')
	return parser.parse_args()


if __name__=='__main__':

	args = init_arg()
	if not os.path.exists("./tmp_models/"):
		os.mkdir("./tmp_models/")
	
	multistep = str(args.multistep) == "True"

	logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
	logging.getLogger().setLevel(logging.INFO)


	logging.info("WANDB init...")

	# start a new run
	run = wandb.init(
		project="TimePFN_run",
		entity=wandb_entity,
		config=f"./src/config/{args.experiment}.yml"
	)

	config = wandb.config

	logging.info("Generating dataset")
	pickle_map = get_cancer_sim_data(
		chemo_coeff=args.chemo_coeff,
		radio_coeff=args.radio_coeff,
		b_load=True,
		b_save=False,
		model_root=args.results_dir,
	)

	wandb.log({"chemo_coeff": args.chemo_coeff})
	wandb.log({"radio_coeff":args.radio_coeff})

	kappa = int(args.kappa)
	wandb.log({"kappa":kappa})

	lambda_val = float(args.lambda_val)
	wandb.log({"lambda":lambda_val})

	max_samples = int(args.max_samples)
	wandb.log({"max_samples":max_samples})

	max_horizon = int(args.max_horizon)
	wandb.log({"max_horizon":max_horizon})

	strategy = "all"
	wandb.log({"strategy":strategy})

	coeff = int(args.radio_coeff)

	logging.info("Transforming dataset")
	pickle_map = transform_data(
		data=pickle_map,
		interpolate=False,
		strategy=strategy, 
		sample_prop=1,
		kappa=kappa,
		max_samples=max_samples
	)

	logging.info("Processing dataset")
	training_processed, validation_processed, test_processed = process_data(pickle_map)

	use_time = False 


	logging.info("Training model..")
	TimePFN_trainer = trainer(
		run=run,
		args=args,
		lambda_val=args.lambda_val,
	)

	TimePFN_trainer.fit(
		train_data=training_processed,
		validation_data=validation_processed,
		epochs=config["epochs"],
		patience=config["patience"],
		batch_size=config["batch_size"]
	)
	
	logging.info("Testing model...")
	TimePFN_trainer.predict(test_data=test_processed)

	
	run.finish()

	os.system("rm -rf ./tmp_models/")
	