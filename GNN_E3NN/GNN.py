import numpy as np
import optuna
import pickle as pkl
from collections import defaultdict

import torch
import torch_geometric
from torch.optim import Adam, AdamW

from models import PeriodicNetwork

from weights_updater import getWeightUpdater

from dataset import loadDataset

from utils import (
    get_logger, denormalizeLoss, numpy, stackingTensorsToNumpy,
    getSequentialInput, splitTrainValTest, saveGNNResults,
)

class GNN():
    def __init__(self, dataset, num_layers, num_hidden_channels, multiplicity, lmax=2,
                 r_max=6, lr=5e-3, weight_decay=0.05, dropout=0, batchsz=128, max_epoch=100):
        torch_geometric.seed_everything(4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset_metadata = dataset.metadata
        self.loaders = splitTrainValTest(dataset=dataset, batchsz=batchsz)

        data = dataset[0]
        in_dim = data.x.shape[1]

        self.out_dim = [1, 251, 91]
        self.model = PeriodicNetwork(
            in_dim=in_dim,
            em_dim=num_hidden_channels,
            irreps_in=str(num_hidden_channels)+"x0e",
            irreps_out=str(self.out_dim[0])+"x0e+" + str(self.out_dim[1])+"x0e+" + str(self.out_dim[2])+"x0e",
            irreps_node_attr=str(num_hidden_channels)+"x0e",
            layers=num_layers,
            mul=multiplicity,
            lmax=lmax,
            max_radius=r_max,
            num_neighbors=63,
            reduce_output=True,
        )
        self.model.to(self.device)
        self.name = f'E3NN_{num_layers}_{num_hidden_channels}_{multiplicity}_{lmax}_' \
                    f'{lr}_{weight_decay}_{dropout}_{batchsz}'

        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20, T_mult=1,
            eta_min=0
        )

        self.max_epoch = max_epoch
        self.logger = get_logger(f'./logfiles/{self.name}')

    def lossFunction(self, out, target, MSE=False):
        # return torch.sum(torch.square(out - y))
        return torch.sum(torch.abs(out - target))
        # return torch.nn.SmoothL1Loss(reduction='sum')(out, target)

    def trainModel(self, model, loader):
        model.train()
        loss_cumulative, loss_cumulative_mae = 0, 0

        for batch in loader:
            batch = batch.to(self.device)

            output = model(batch)
            output1 = output[:, : self.out_dim[0]]
            output2 = output[:, self.out_dim[0] : self.out_dim[0] + self.out_dim[1]]
            output3 = output[:, self.out_dim[0] + self.out_dim[1] : ]

            y1, y2, y3 = batch.energy, batch.optcond, batch.elecond

            loss1 = torch.nn.MSELoss()(output1, y1)
            loss2 = torch.nn.MSELoss()(output2, y2)
            loss3 = torch.nn.MSELoss()(output3, y3)
            loss = loss1 + loss2 + loss3

            # Compute MAEs
            loss1_mae = torch.nn.L1Loss()(output1, y1)
            loss2_mae = torch.nn.L1Loss()(output2, y2)
            loss3_mae = torch.nn.L1Loss()(output3, y3)
            loss_mae = loss1_mae + loss2_mae + loss3_mae

            loss_cumulative += loss.item()
            loss_cumulative_mae += loss_mae.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss_cumulative / len(loader), loss_cumulative_mae / len(loader)

    def evalModel(self, model, loader, write_output=False):
        model.eval()

        DFT, GNN = defaultdict(list), defaultdict(list)
        loss_cumulative, loss_cumulative_mae = 0, 0

        for batch in loader:
            batch = batch.to(self.device)

            output = model(batch)
            output1 = output[:, : self.out_dim[0]]
            output2 = output[:, self.out_dim[0] : self.out_dim[0] + self.out_dim[1]]
            output3 = output[:, self.out_dim[0] + self.out_dim[1] : ]

            y1, y2, y3 = batch.energy, batch.optcond, batch.elecond

            loss1 = torch.nn.MSELoss()(output1, y1)
            loss2 = torch.nn.MSELoss()(output2, y2)
            loss3 = torch.nn.MSELoss()(output3, y3)
            loss = loss1 + loss2 + loss3

            # Compute MAEs
            loss1_mae = torch.nn.L1Loss()(output1, y1)
            loss2_mae = torch.nn.L1Loss()(output2, y2)
            loss3_mae = torch.nn.L1Loss()(output3, y3)
            loss_mae = loss1_mae + loss2_mae + loss3_mae

            loss_cumulative += loss.item()
            loss_cumulative_mae += loss_mae.item()
        return loss_cumulative / len(loader), loss_cumulative_mae / len(loader)

    def run(self):
        min_val_loss = float('inf')
        early_schedule_step = 0
        history = []

        train_loader, val_loader, test_loader = self.loaders['train'], self.loaders['val'], self.loaders['test']

        # self.logger.info('')
        for epoch in range(self.max_epoch):
            train_loss, train_loss_mae = self.trainModel(self.model, train_loader)
            val_loss, val_loss_mae = self.evalModel(self.model, val_loader)

            history.append({
                'epoch': epoch,
                'train': {'loss_mse': train_loss, 'loss_mae': train_loss_mae},
                'val': {'loss_mse': val_loss, 'loss_mae': val_loss_mae},
            })
            self.logger.info(f'epoch = {epoch}, training loss = {train_loss}, validation loss = {val_loss}')

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            if epoch == 1 or val_loss < min_val_loss:
                min_val_loss = val_loss
                early_schedule_step = 0
                # saveTrainedModel(self.name, self.model)
                torch.save(self.model.state_dict(), f'./save/{self.name}_model.torch')
            else:
                early_schedule_step += 1
                self.logger.info(f'Early stopping step {early_schedule_step}, the current validation loss {val_loss}'
                                 f' is larger than best value {min_val_loss}.')

            # if self.early_schedule_step == 8:
            #     self.logger.info('Early stopped at epoch {}'.format(epoch))
            #     break

        self.model.load_state_dict(torch.load(f'./save/{self.name}_model.torch'))
        # best_model = loadTrainedModel(self.name).to(self.device)
        # test_loss, combine_test_loss, test_output = self.evalModel(best_model, test_loader, write_output=True)
        test_loss, test_loss_mae = self.evalModel(self.model, test_loader)
        self.logger.info('=' * 100)
        self.logger.info(f'The testing loss is {test_loss}.')
        saveGNNResults(
            name=self.name,
            dataset_metadata=self.dataset_metadata,
            history=history,
            test_loss=test_loss,
        )
        return test_loss, test_loss_mae

class GNN_optuna():
    def __init__(self, dataset, num_trials, max_epoch):
        self.dataset = dataset
        self.num_trials = num_trials
        self.max_epoch = max_epoch

    def save(self, study):
        with open('./save/GNN_optuna.pkl', 'wb') as f:
            pkl.dump(study, f)

    def objective(self, trial):
        num_layers = trial.suggest_int('num_layers', 1, 3)
        num_hidden_channels = trial.suggest_categorical('num_hidden_channels', [8, 16, 32, 64, 128])
        multiplicity = trial.suggest_categorical('multiplicity', [2, 4, 8, 16, 32])
        lr = trial.suggest_float('lr', 1e-4, 1e-2)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1)
        batchsz = trial.suggest_int('batchsz', 2, 16)

        GNN_trial = GNN(
            dataset=dataset,
            num_layers=num_layers,
            num_hidden_channels=num_hidden_channels,
            multiplicity=multiplicity,
            lmax=2,
            r_max=6,
            lr=lr,
            weight_decay=weight_decay,
            batchsz=batchsz,
            max_epoch=self.max_epoch,
        )
        test_loss = GNN_trial.run()[1]
        return test_loss

    def run(self):
        study = optuna.create_study(directions=['minimize'])
        study.optimize(self.objective, n_trials=self.num_trials)

        print("Best trial:")
        trial = study.best_trial
        print("Value: ", trial.value)
        print("Params: ")
        for key, value in trial.params.items(): print("{}: {}".format(key, value))

        self.save(study)

if __name__ == '__main__':
    torch_geometric.seed_everything(4)
    normalization_func = 'mean'
    axis = 'all'
    dataset_root = f'datasets/CGCNN/{normalization_func}/{axis}'
    dataset = loadDataset(dataset_root, atomic_feature_type='CGCNN', normalization_func=normalization_func, axis=axis)

    # gnn = GNN(
    #     dataset=dataset,
    #     num_layers=2,
    #     num_hidden_channels=64,
    #     multiplicity=32,
    #     lmax=2,
    #     r_max=6,
    #     lr=5e-3,
    #     weight_decay=0.05,
    #     batchsz=4,
    #     max_epoch=100,
    # )
    # gnn.run()

    gnn_optuna = GNN_optuna(
        dataset=dataset,
        num_trials=100,
        max_epoch=5,
    )
    gnn_optuna.run()