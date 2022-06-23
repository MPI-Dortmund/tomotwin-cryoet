"""
MIT License

Copyright (c) 2021 MPI-Dortmund

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
from typing import Any, Dict, Tuple, Iterable
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch import nn
from torch.utils.data import DataLoader
import torch
from torch import optim
from torch.backends import cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

from tomotwin.modules.training.trainer import Trainer
from tomotwin.modules.training.tripletdataset import TripletDataset
from tomotwin.modules.networks.torchmodel import TorchModel
from tomotwin.modules.common import preprocess




class TorchTrainer(Trainer):
    """
    Trainer for pytorch.
    """

    def __init__(
        self,
        epochs: int,
        batchsize: int,
        learning_rate: float,
        network: TorchModel,
        criterion: nn.Module,
        training_data: TripletDataset = None,
        test_data: TripletDataset = None,
        workers: int = 0,
        output_path: str = None,
        log_dir: str = None,
        checkpoint: str = None,
        optimizer: str = "Adam",
        amsgrad: bool = False,
        weight_decay: float = 0,
        patience: int = None,
        save_epoch_seperately: bool = False
    ):
        """
        :param epochs: Number of epochs
        :param batchsize: Training batch size
        :param learning_rate: The learning rate
        """

        super().__init__()
        cudnn.benchmark = True
        self.epochs = epochs
        self.batchsize = batchsize
        self.learning_rate = learning_rate
        self.training_data = training_data
        self.test_data = test_data
        self.patience = patience
        if self.patience is None:
            self.patience = self.epochs
        self.workers = workers
        self.best_model_loss = None
        self.best_model_f1 = None
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.criterion = criterion
        self.network = network
        self.network_config = None
        self.output_path = output_path
        self.last_loss = None
        self.best_val_loss = np.Infinity

        self.best_val_f1 = 0
        self.current_epoch = None
        self.best_epoch_loss = None
        self.best_epoch_f1 = None
        self.checkpoint = None
        self.start_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.init_weights()
        self.model = self.network.get_model()
        self.checkpoint = checkpoint
        self.save_epoch_seperately = save_epoch_seperately
        self.f1_improved = False
        self.loss_improved = False



        # Write graph to tensorboard
        dummy_input = torch.zeros([12, 1, 37, 37, 37])
        self.writer.add_graph(self.model, dummy_input)

        self.model = self.model.to(self.device)
        self.optimizer = getattr(optim, optimizer)(
            self.model.parameters(),
            lr=self.learning_rate,
            amsgrad=amsgrad,
            weight_decay=weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=patience, verbose=True
        )
        model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_params])
        print("Number of parameters:", params)

        self.writer.add_text("Optimizer", type(self.optimizer).__name__)
        self.writer.add_text("Initial learning rate", str(self.learning_rate))

        if self.checkpoint is not None:
            self.load_checkpoint(checkpoint=self.checkpoint)

        self.model = nn.DataParallel(self.model)

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.seed()

    def get_train_test_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.training_data,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=False,
            # prefetch_factor=5,
            timeout=60,
        )

        test_loader = None
        if self.test_data is not None:
            test_loader = DataLoader(
                self.test_data,
                batch_size=self.batchsize,
                shuffle=True,
                num_workers=self.workers,
                pin_memory=False,
                # prefetch_factor=5,
                timeout=60,
            )
        return train_loader, test_loader

    @staticmethod
    def get_best_f1(
        anchor_label: str, similarities: np.array, sim_labels: Iterable
    ) -> Tuple[float, float]:
        PDB = os.path.splitext(anchor_label)[0].upper()
        gt_mask = np.array([PDB in p.upper() for p in sim_labels])
        best_f1 = 0
        best_t = None
        for t in np.arange(0, 1, 0.025):
            picked = similarities > t

            true_positive = np.logical_and(gt_mask, picked)
            TP = np.sum(true_positive)
            false_positive = np.logical_and(gt_mask == False, picked)
            FP = np.sum(false_positive)
            false_negative = np.logical_and(gt_mask, picked == False)
            FN = np.sum(false_negative)
            f1 = 2 * TP / (2 * TP + FP + FN)
            if f1 >= best_f1:
                best_t = t
                best_f1 = f1

        return best_f1, best_t

    @staticmethod
    def calc_avg_f1(anchors: pd.DataFrame, volumes: pd.DataFrame) -> float:
        """
        Calculates average f1 score
        Each column in 'anchors' represents an anchor volume.
        Each column in 'volumes' represents an tomogram subvolume
        :return: Classification accuracy
        """
        scores = []
        for col in anchors:
            sim = np.matmul(volumes.T, anchors[col])
            best_f1, best_t = TorchTrainer.get_best_f1(
                anchor_label=col, similarities=sim, sim_labels=sim.index.values
            )
            scores.append(best_f1)
        avg_f1 = np.mean(scores)
        return avg_f1

    def classification_f1_score(self, test_loader: DataLoader) -> float:
        self.model.eval()
        t = tqdm(test_loader, desc="Classification accuracy", leave=False)
        anchor_emb = {}  # pd.DataFrame()
        vol_emb = {}  # pd.DataFrame()

        with torch.no_grad():
            for _, batch in enumerate(t):
                anchor_vol = batch["anchor"].to(self.device, non_blocking=True)
                positive_vol = batch["positive"].to(self.device, non_blocking=True)
                negative_vol = batch["negative"].to(self.device, non_blocking=True)
                filenames = batch["filenames"]
                with autocast():
                    anchor_out = self.model.forward(anchor_vol)
                    positive_out = self.model.forward(positive_vol)
                    negative_out = self.model.forward(negative_vol)

                    anchor_out_np = anchor_out.cpu().detach().numpy()
                    for i, anchor_filename in enumerate(filenames[0]):
                        if preprocess.label_filename(anchor_filename) not in anchor_emb:
                            anchor_emb[
                                preprocess.label_filename(anchor_filename)
                            ] = anchor_out_np[i, :]
                    positive_out_np = positive_out.cpu().detach().numpy()
                    for i, pos_filename in enumerate(filenames[1]):
                        if os.path.basename(pos_filename) not in vol_emb:
                            vol_emb[os.path.basename(pos_filename)] = positive_out_np[
                                i, :
                            ]

                    negative_out_np = negative_out.cpu().detach().numpy()
                    for i, neg_filename in enumerate(filenames[2]):
                        if os.path.basename(neg_filename) not in vol_emb:
                            vol_emb[os.path.basename(neg_filename)] = negative_out_np[
                                i, :
                            ]

        return TorchTrainer.calc_avg_f1(pd.DataFrame(anchor_emb), pd.DataFrame(vol_emb))

    def run_batch(self, batch):
        anchor_vol = batch["anchor"].to(self.device, non_blocking=True)
        positive_vol = batch["positive"].to(self.device, non_blocking=True)
        negative_vol = batch["negative"].to(self.device, non_blocking=True)
        with autocast():
            anchor_out = self.model.forward(anchor_vol)
            positive_out = self.model.forward(positive_vol)
            negative_out = self.model.forward(negative_vol)

            loss = self.criterion(
                anchor_out,
                positive_out,
                negative_out,
                label_anchor=batch["label_anchor"],
                label_positive=batch["label_positive"],
                label_negative=batch["label_negative"],
            )
        return loss

    def save_best_loss(self, current_val_loss: float, epoch: int) -> None:
        if current_val_loss < self.best_val_loss:
            self.loss_improved = True
            print(
                f"Validation loss improved from {self.best_val_loss} to {current_val_loss}"
            )
            self.best_epoch_loss = epoch
            self.best_val_loss = current_val_loss
            self.best_model_loss = copy.deepcopy(self.model)

    def save_best_f1(self, current_val_f1: float, epoch: int) -> None:
        if current_val_f1 > self.best_val_f1:
            self.f1_improved = True
            print(
                f"Validation F1 score improved from {self.best_val_f1} to {current_val_f1}"
            )
            self.best_epoch_f1 = epoch
            self.best_val_f1 = current_val_f1
            self.best_model_f1 = copy.deepcopy(self.model)

    def validation_loss(self, test_loader: DataLoader) -> float:
        """
        Runs the current model on the validation data
        :return: Validation loss
        """
        val_loss = []
        self.model.eval()
        t = tqdm(test_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for _, batch in enumerate(t):
                valloss = self.run_batch(batch)
                val_loss.append(valloss.cpu().detach().numpy())
                desc_t = f"Validation (running loss: {np.mean(val_loss[-20:]):.4f} "
                t.set_description(desc=desc_t)

        current_val_loss = np.mean(val_loss)
        return current_val_loss

    def load_checkpoint(self, checkpoint: str) -> None:
        """
        :param checkpoint: Path to checkpoint
        :return:
        """

        try:
            self.checkpoint = torch.load(checkpoint)
        except FileNotFoundError:
            print(f"Checkpoint {checkpoint} can't be found. Ignore it.")
            self.checkpoint = None
            return

        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
        self.start_epoch = self.checkpoint["epoch"] + 1
        self.last_loss = self.checkpoint["loss"]
        self.best_val_loss = self.checkpoint["best_loss"]
        self.best_val_f1 = self.checkpoint["best_f1"]
        print(
            f"Restart from checkpoint. Epoch: {self.start_epoch}, Training loss: {self.last_loss}, Validation loss: {self.best_val_loss}"
        )

    def epoch(self, train_loader: DataLoader) -> float:
        """
        Runs a single epoch
        :param train_loader: Data loader for training data
        :return: Training loss after the epoch
        """

        scaler = GradScaler()
        running_loss = []
        self.model.train()
        t = tqdm(train_loader, desc="Training", leave=False)
        for _, batch in enumerate(t):
            self.optimizer.zero_grad()

            loss = self.run_batch(batch)
            loss_np = loss.cpu().detach().numpy()
            running_loss.append(loss_np)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            desc_t = f"Training (loss: {np.mean(running_loss[-20:]):.4f} "

            t.set_description(desc=desc_t)

        training_loss = np.mean(running_loss)
        self.last_loss = training_loss
        return training_loss

    def train(self) -> nn.Module:
        """
        Trains the model and returns it.
        :return: Trained model
        """
        if self.training_data is None:
            raise RuntimeError("Training data is not set")

        train_loader, test_loader = self.get_train_test_dataloader()

        # Training Loop
        for epoch in tqdm(
            range(self.start_epoch, self.epochs),
            initial=self.start_epoch,
            total=self.epochs,
            desc="Epochs",
        ):
            self.f1_improved = False
            self.loss_improved = False
            self.current_epoch = epoch
            train_loss = self.epoch(train_loader=train_loader)

            print(
                "Epoch: {}/{} - Training Loss: {:.4f}".format(
                    epoch + 1, self.epochs, train_loss
                )
            )
            self.writer.add_scalar("Loss/train", train_loss, epoch)

            # Validation
            if test_loader is not None:
                current_val_loss = self.validation_loss(test_loader)
                current_val_f1 = self.classification_f1_score(test_loader=test_loader)
                self.scheduler.step(current_val_loss)
                print(f"Validation Loss: {current_val_loss:.4f}.")
                print(f"Validation F1 Score: {current_val_f1:.4f}.")
                self.writer.add_scalar("Loss/validation", current_val_loss, epoch)
                self.writer.add_scalar("F1/validation", current_val_f1, epoch)
                self.save_best_loss(current_val_loss, epoch)
                self.save_best_f1(current_val_f1, epoch)

            self.writer.flush()

            if self.output_path is not None:
                self.write_results_to_disk(self.output_path, save_each_improvement=self.save_epoch_seperately)

        return self.model

    def set_training_data(self, training_data: TripletDataset) -> None:
        self.training_data = training_data

    def set_test_data(self, test_data: TripletDataset) -> None:
        self.test_data = test_data

    def set_network_config(self, config):
        self.network_config = config

    @staticmethod
    def _write_model(
        path: str,
        model: TorchModel,
        config: Dict,
        optimizer=None,
        loss: float = None,
        epoch: int = None,
        best_loss: float = None,
        best_f1: float = None,
    ):
        """
        Adds some metadata to the model and write the model  to disk

        :param path: Path where the model should be written
        :param model: The model that is saved to disk
        :param config: Configuration of tomotwin
        :param optimizer: Optimizer
        :param loss: Loss
        :param epoch: Current epoch
        :param best_loss: Current best validation loss
        :param best_f1:  Current best validtion f1 score
        :return:
        """
        results_dict = {
            "model_state_dict": model.state_dict(),
            "tomotwin_config": config,
        }
        if optimizer is not None:
            results_dict["optimizer_state_dict"] = optimizer.state_dict()

        if loss is not None:
            results_dict["loss"] = loss

        if best_loss is not None:
            results_dict["best_loss"] = best_loss

        if best_f1 is not None:
            results_dict["best_f1"] = best_f1

        if epoch is not None:
            results_dict["epoch"] = epoch

        torch.save(
            results_dict,
            path,
        )

    def write_model_to_disk(self, path: str, model_to_save, model_name: str, epoch: int):
        if isinstance(model_to_save, nn.DataParallel):
            model_to_save = model_to_save.module

        self._write_model(
            path=os.path.join(path, model_name),
            model=model_to_save,
            config=self.network_config,
            optimizer=self.optimizer,
            loss=self.last_loss,
            best_loss=self.best_val_loss,
            best_f1=self.best_val_f1,
            epoch=epoch,
        )

    def write_results_to_disk(self, path: str, save_each_improvement: bool = False):

        self.write_model_to_disk(path, self.model, "latest.pth", self.current_epoch)

        if self.current_epoch == self.epochs - 1:
            if os.path.exists(os.path.join(path, "final.pth")):
                os.remove(os.path.join(path, "final.pth"))
            os.rename(os.path.join(path, "latest.pth"), os.path.join(path, "final.pth"))

        if self.best_model_loss is not None:
            # The best_model can be None, after a training restart.
            self.write_model_to_disk(path, self.best_model_loss, "best_loss.pth", self.best_epoch_loss)

        if self.best_model_f1 is not None:
            # The best_model can be None, after a training restart.
            self.write_model_to_disk(path, self.best_model_f1, "best_f1.pth", self.best_epoch_f1)

        if save_each_improvement and self.f1_improved:
            self.write_model_to_disk(path, self.best_model_f1, f"best_f1_{self.best_epoch_f1}.pth", self.best_epoch_f1)


    def get_model(self) -> Any:
        return self.model
