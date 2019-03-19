#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import make_grid
from base.base_trainer import BaseTrainer


#------------------------------------------------------------------------------
#  Poly learning-rate Scheduler
#------------------------------------------------------------------------------
def poly_lr_scheduler(optimizer, init_lr, curr_iter, max_iter, power=0.9):
	for g in optimizer.param_groups:
		g['lr'] = init_lr * (1 - curr_iter/max_iter)**power


#------------------------------------------------------------------------------
#   Class of Trainer
#------------------------------------------------------------------------------
class Trainer(BaseTrainer):
	"""
	Trainer class

	Note:
		Inherited from BaseTrainer.
	"""
	def __init__(self, model, loss, metrics, optimizer, resume, config,
				 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
		super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
		self.config = config
		self.data_loader = data_loader
		self.valid_data_loader = valid_data_loader
		self.do_validation = self.valid_data_loader is not None
		self.lr_scheduler = lr_scheduler
		self.max_iter = len(self.data_loader) * self.epochs
		self.init_lr = optimizer.param_groups[0]['lr']


	def _eval_metrics(self, output, target):
		acc_metrics = np.zeros(len(self.metrics))
		for i, metric in enumerate(self.metrics):
			acc_metrics[i] += metric(output, target)
		return acc_metrics


	def _train_epoch(self, epoch):
		"""
		Training logic for an epoch

		:param epoch: Current training epoch.
		:return: A log that contains all information you want to save.

		Note:
			If you have additional information to record, for example:
				> additional_log = {"x": x, "y": y}
			merge it with log before return. i.e.
				> log = {**log, **additional_log}
				> return log

			The metrics in log must have the key 'metrics'.
		"""
		print("Train on epoch...")
		self.model.train()
		self.writer_train.set_step(epoch)
	
		# Perform training
		total_loss = 0
		total_metrics = np.zeros(len(self.metrics))
		n_iter = len(self.data_loader)
		for batch_idx, (data, target) in tqdm(enumerate(self.data_loader), total=n_iter):
			curr_iter = batch_idx + (epoch-1)*n_iter
			data, target = data.to(self.device), target.to(self.device)
			self.optimizer.zero_grad()
			output = self.model(data)
			loss = self.loss(output, target)
			loss.backward()
			self.optimizer.step()

			total_loss += loss.item()
			total_metrics += self._eval_metrics(output, target)

			if (batch_idx==n_iter-2) and (self.verbosity>=2):
				self.writer_train.add_image('train/input', make_grid(data[:,:3,:,:].cpu(), nrow=4, normalize=True))
				self.writer_train.add_image('train/label', make_grid(target.unsqueeze(1).cpu(), nrow=4, normalize=True))
				if type(output)==tuple or type(output)==list:
					self.writer_train.add_image('train/output', make_grid(F.softmax(output[0], dim=1)[:,1:2,:,:].cpu(), nrow=4, normalize=True))
				else:
					# self.writer_train.add_image('train/output', make_grid(output.cpu(), nrow=4, normalize=True))
					self.writer_train.add_image('train/output', make_grid(F.softmax(output, dim=1)[:,1:2,:,:].cpu(), nrow=4, normalize=True))

			poly_lr_scheduler(self.optimizer, self.init_lr, curr_iter, self.max_iter, power=0.9)

		# Record log
		total_loss /= len(self.data_loader)
		total_metrics /= len(self.data_loader)
		log = {
			'train_loss': total_loss,
			'train_metrics': total_metrics.tolist(),
		}

		# Write training result to TensorboardX
		self.writer_train.add_scalar('loss', total_loss)
		for i, metric in enumerate(self.metrics):
			self.writer_train.add_scalar('metrics/%s'%(metric.__name__), total_metrics[i])

		if self.verbosity>=2:
			for i in range(len(self.optimizer.param_groups)):
				self.writer_train.add_scalar('lr/group%d'%(i), self.optimizer.param_groups[i]['lr'])

		# Perform validating
		if self.do_validation:
			print("Validate on epoch...")
			val_log = self._valid_epoch(epoch)
			log = {**log, **val_log}

		# Learning rate scheduler
		if self.lr_scheduler is not None:
			self.lr_scheduler.step()

		return log


	def _valid_epoch(self, epoch):
		"""
		Validate after training an epoch

		:return: A log that contains information about validation

		Note:
			The validation metrics in log must have the key 'valid_metrics'.
		"""
		self.model.eval()
		total_val_loss = 0
		total_val_metrics = np.zeros(len(self.metrics))
		n_iter = len(self.valid_data_loader)
		self.writer_valid.set_step(epoch)

		with torch.no_grad():
			# Validate
			for batch_idx, (data, target) in tqdm(enumerate(self.valid_data_loader), total=n_iter):
				data, target = data.to(self.device), target.to(self.device)
				output = self.model(data)
				loss = self.loss(output, target)

				total_val_loss += loss.item()
				total_val_metrics += self._eval_metrics(output, target)

				if (batch_idx==n_iter-2) and(self.verbosity>=2):
					self.writer_valid.add_image('valid/input', make_grid(data[:,:3,:,:].cpu(), nrow=4, normalize=True))
					self.writer_valid.add_image('valid/label', make_grid(target.unsqueeze(1).cpu(), nrow=4, normalize=True))
					if type(output)==tuple or type(output)==list:
						self.writer_valid.add_image('valid/output', make_grid(F.softmax(output[0], dim=1)[:,1:2,:,:].cpu(), nrow=4, normalize=True))
					else:
						# self.writer_valid.add_image('valid/output', make_grid(output.cpu(), nrow=4, normalize=True))
						self.writer_valid.add_image('valid/output', make_grid(F.softmax(output, dim=1)[:,1:2,:,:].cpu(), nrow=4, normalize=True))

			# Record log
			total_val_loss /= len(self.valid_data_loader)
			total_val_metrics /= len(self.valid_data_loader)
			val_log = {
				'valid_loss': total_val_loss,
				'valid_metrics': total_val_metrics.tolist(),
			}

			# Write validating result to TensorboardX
			self.writer_valid.add_scalar('loss', total_val_loss)
			for i, metric in enumerate(self.metrics):
				self.writer_valid.add_scalar('metrics/%s'%(metric.__name__), total_val_metrics[i])

		return val_log