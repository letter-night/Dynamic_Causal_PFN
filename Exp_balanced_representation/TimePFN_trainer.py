import logging
import numpy as np
import torch
import torch.nn as nn
import torchcde 
import torchmetrics
from torch.nn import functional as F
from tqdm import tqdm

from src.models.TimePFN import TimePFN
from src.utils.data_utils import (
	data_to_torch_tensor,
)

from src.utils.losses import compute_norm_mse_loss
from src.utils.training_tools import EarlyStopping


class trainer:
	def __init__(
			self, 
			run,
			args,
			lambda_val=None,
	):
		self.run=run

		self.model = None
		self.device=None

		self.args=args 

		self.lambda_val = lambda_val

	
	def _train(self, train_dataloader, model, optimizer, lambda_val):
		model = model.train()

		treatment_loss = nn.CrossEntropyLoss()

		train_losses_total = []
		train_losses_y = []
		train_losses_a = []

		for (batch_coeffs_x, batch_y, batch_treat) in train_dataloader:

			batch_coeffs_x = torch.tensor(batch_coeffs_x, dtype=torch.float, device=self.device)

			outcomes = torch.tensor(batch_y[:, :, 0], dtype=torch.float, device=self.device)
			active_entries = torch.tensor(batch_y[:, :, 1], dtype=torch.float, device=self.device)

			current_treatment = torch.tensor(batch_treat[:, -1, :], dtype=torch.float, device=self.device)

			pred_y, pred_a = model(batch_coeffs_x, self.device)

			loss_y = compute_norm_mse_loss(outcomes, pred_y, active_entries)

			loss_a = treatment_loss(pred_a, torch.argmax(current_treatment, dim=1))

			total_loss = loss_y + lambda_val * loss_a 

			total_loss.backward()

			optimizer.step()
			optimizer.zero_grad()

			train_losses_total.append(total_loss.item())
			train_losses_y.append(loss_y.item())
			train_losses_a.append(loss_a.item())
		
		return model, train_losses_total, train_losses_y, train_losses_a
	

	def _test(self, test_dataloader, model, lambda_val):
		model = model.eval()

		treatment_loss = nn.CrossEntropyLoss()

		test_losses_total = []
		test_losses_y = []
		test_losses_a = []

		with torch.no_grad():
			for (batch_coeffs_x_val, batch_y_val, batch_treat_val) in test_dataloader:

				batch_coeffs_x_val = torch.tensor(batch_coeffs_x_val, dtype=torch.float, device=self.device)

				outcomes_val = torch.tensor(batch_y_val[:, :, 0], dtype=torch.float, device=self.device)
				active_entries_val = torch.tensor(batch_y_val[:, :, 1], dtype=torch.float, device=self.device)

				current_treatment_val = torch.tensor(batch_treat_val[:, -1, :], dtype=float, device=self.device)

				pred_y_val, pred_a = model(batch_coeffs_x_val, self.device)

				loss_y_val = compute_norm_mse_loss(outcomes_val, pred_y_val, active_entries_val)

				loss_a_val = treatment_loss(pred_a, torch.argmax(current_treatment_val, dim=1))

				total_loss_val = loss_y_val + lambda_val * loss_a_val

				test_losses_total.append(total_loss_val.item())
				test_losses_y.append(loss_y_val.item())
				test_losses_a.append(loss_a_val.item())
		
		return model, test_losses_total, test_losses_y, test_losses_a
	

	def prepare_dataloader(self, data, batch_size):
		data_X, data_A, _, data_y, data_tr, _, _ = data_to_torch_tensor(
			data, 
			sample_prop=1
		)

		data_concat = torch.cat((data_X, data_A), 2)

		data_shape = list(data_concat.shape)

		# Need to check!! (we interpolate because of irregular sampling)
		coeffs = torchcde.linear_interpolation_coeffs(data_concat)

		dataset = torch.utils.data.TensorDataset(coeffs, data_y, data_tr)
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

		return dataloader, data_shape	
	

	def fit(self, train_data, validation_data, epochs, patience, batch_size):
		logging.info("Getting training data")

		if torch.cuda.is_available():
			device_type="cuda"
		else:
			device_type="cpu"
		
		device = torch.device(device_type)

		self.device=device

		logging.info(f"Predicting using device: {device_type}")
		early_stopping = EarlyStopping(patience=patience, delta=0.0001)

		# create dataloaders
		train_dataloader, data_shape = self.prepare_dataloader(
			data=train_data,
			batch_size=int(batch_size)
		)
		val_dataloader, _ = self.prepare_dataloader(
			data=validation_data,
			batch_size=int(batch_size)
		)

		logging.info("Instantiating TimePFN")

		model = TimePFN(self.args)

		model = model.to(self.device)

		optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

		self.run.watch(model, log="all")

		logging.info("Training TimePFN")
		epochs=100

		lambda_vals = np.linspace(0.0, 1.0, num=epochs)

		for epoch in tqdm(range(epochs)):

			lambda_val = lambda_vals[epoch]
			logging.info(f"Training epoch: {epoch} & lambda: {lambda_val}")

			model, train_losses_total, train_losses_y, train_losses_a = self._train(
				train_dataloader=train_dataloader,
				model=model,
				optimizer=optimizer,
				lambda_val=lambda_val
			)

			logging.info(f"Validation epoch: {epoch}")
			model, val_losses_total, val_losses_y, val_losses_a = self._test(
				test_dataloader=val_dataloader,
				model=model,
				lambda_val=lambda_val
			)

			tqdm.write(
                f"Epoch: {epoch}   Training loss: {np.average(train_losses_total)} ; Train Treatment loss: {np.average(train_losses_a)} ; Train Outcome loss: {np.average(train_losses_y)}, Val loss: {np.average(val_losses_total)} ;Val Treatment loss: {np.average(val_losses_a)} ; Val Outcome loss: {np.average(val_losses_y)}",
            )

			if int(np.average(train_losses_y)) > 100000 or np.average(
                train_losses_y,
            ) == float("nan"):
				import sys

				raise ValueError("Exiting run...")
			
			self.run.log(
				{
					"Epoch": epoch,
                    "Training loss": np.average(train_losses_total),
                    "Train Treatment loss": np.average(train_losses_a),
                    "Train Outcome loss": np.average(train_losses_y),
                    "Val loss": np.average(val_losses_total),
                    "Val Treatment loss": np.average(val_losses_a),
                    "Val Outcome loss": np.average(val_losses_y),
				}
			)

			torch.save(model.state_dict(), f"./tmp_models/model_epoch_{epoch}.h5")
			self.run.save(f"./tmp_models/model_epoch_{epoch}.h5")

			early_stopping(np.average(val_losses_y), model)

			if early_stopping.early_stop:
				print("Early stopping phases initiated...")
				break
		
		model.load_state_dict(torch.load("checkpoint.pt"))
		
		torch.save(model.state_dict(), "./tmp_models/model_final.h5")
		self.run.save("./tmp_models/model_final.h5")

		self.model=model 

	
	def predict(self, test_data):

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		logging.info(f"Predicting with: {device}")

		treatment_loss = nn.CrossEntropyLoss()

		(test_X, test_A, _, test_y, test_treat, _, _) = data_to_torch_tensor(
			test_data, sample_prop=1
		)

		test_concat = torch.cat((test_X, test_A), 2)

		# Need check!!!!
		test_coeffs = torchcde.linear_interpolation_coeffs(test_concat)
		# test_coeffs = torch.tensor(test_coeffs, dtype=torch.float, device=device)
		
		dataset = torch.utils.data.TensorDataset(test_coeffs, test_y, test_treat)
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

		losses_y = []
		losses_a = []
		# accuracy = []

		with torch.no_grad():
			for (batch_x_coeffs_test, batch_y_test, batch_tr_test) in dataloader:

				batch_x_coeffs_test = torch.tensor(batch_x_coeffs_test, dtype=torch.float, device=self.device)

				outcomes_test = torch.tensor(batch_y_test[:, :, 0], dtype=torch.float, device=device)
				active_entries_test = torch.tensor(batch_y_test[:, :, 1], dtype=torch.float, device=device)

				current_treatment_test = torch.tensor(batch_tr_test[:, -1, :], dtype=torch.float, device=device)

				pred_y_test, pred_a_test = self.model(batch_x_coeffs_test, device)

				loss_y_test = compute_norm_mse_loss(outcomes_test, pred_y_test, active_entries_test)

				loss_a_test = treatment_loss(pred_a_test, torch.max(current_treatment_test, 1)[1])

				# acc = torchmetrics.functional.accuracy(
				# 	F.softmax(pred_a_test, dim=1),
				# 	torch.argmax(current_treatment_test, dim=1)
				# )

				losses_a.append(loss_a_test.item())
				losses_y.append(loss_y_test.item())
				# accuracy.append(acc.item())


		self.run.log(
			{
				"Test Treatment loss": np.average(losses_a),
                "RMSE Test Outcome loss": np.sqrt(np.average(losses_y)),
                # "Test ACC": np.average(accuracy),
			}
		)
