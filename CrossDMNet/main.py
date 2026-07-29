import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from data_utils.dataloader import get_data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import time
import os
import numpy as np
from configs.config import Config
from sklearn.metrics import accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt
import glob
from models.crossda_mnet import CrossDA_MNet
from loss.center_loss import CenterLoss
from loss.triplet_loss import TripletLoss
import pickle
import torch.nn.functional as F


class Trainer:

    def __init__(self, config):
        self.config = config
        self.data_conf = yaml.safe_load(open('configs' + '/' + self.config.data_name + '.yaml', 'r'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.exp_id = time.strftime("%Y%m%d%H%M")

        LOSO = self.data_conf['LOSO']
        self.result_path = f'results/{self.config.data_name}/{self.config.model_name}/{self.exp_id}'

        os.makedirs(self.result_path, exist_ok=True)

        self.log_writer = open(self.result_path + '/log.txt', 'w')
        self.log_writer.write(f"Model:{self.config.model_name}, DataSet: {self.config.data_name}, Experiment ID: {self.exp_id}\n")

    def init(self):
        self.model = self.build_model().to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.triplet_criterion = TripletLoss(self.device)
        self.centerloss_C = CenterLoss(num_classes=self.data_conf['n_classes'], feat_dim=self.config.n_features).to(self.device)
        self.centerloss_P = CenterLoss(num_classes=self.data_conf['n_classes'], feat_dim=self.config.n_features).to(
            self.device)

        self.optimizer4center_C = optim.SGD(self.centerloss_C.parameters(), lr=0.1)
        self.optimizer4center_P = optim.SGD(self.centerloss_P.parameters(), lr=0.1)

        self.optimizer_C = self.set_optimizer_C()
        self.optimizer_P = self.set_optimizer_P()

        self.central_channels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        self.parietal_channels = torch.tensor([13, 14, 15, 16, 17, 18, 19, 20, 21])

        if self.config.data_name == 'openBMI':
            self.central_channels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            self.parietal_channels = torch.tensor([12, 13, 14, 15, 16, 17, 18, 19])

    def build_model(self):
        chans_C, chans_P = 13, 9
        in_samples = 500
        if self.config.data_name == 'openBMI':
            chans_C, chans_P = 12, 8
        if self.config.data_name == 'GigaDB':
            in_samples = 750
            self.config.n_features = 48 * 13

        model = CrossDA_MNet(
            n_chans=self.data_conf['n_chans'],
            n_cls=self.data_conf['n_classes'],
            in_samples=in_samples,
            chans_C=chans_C,
            chans_P=chans_P
                      )

        return model


    def set_optimizer_C(self):
        conv_params = []
        fc_params = []
        for name, module in self.model.named_modules():
            if 'branchC' in name or 'conv_C' in name or 'classifier_C' in name or 'alpha_f1' in name:
                if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    conv_params.append(module.weight)
                if isinstance(module, nn.Linear):
                    fc_params.append(module.weight)

        return optim.Adam([
            {'params': conv_params, 'weight_decay': self.config.conv_weight_decay},
            {'params': fc_params, 'weight_decay': self.config.fc_weight_decay},
        ], lr=self.config.lr)

    def set_optimizer_P(self):
        conv_params = []
        fc_params = []
        for name, module in self.model.named_modules():
            if 'branchP' in name or 'conv_P' in name or 'classifier_P' in name or 'alpha_f2' in name:
                if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    conv_params.append(module.weight)
                if isinstance(module, nn.Linear):
                    fc_params.append(module.weight)

        return optim.Adam([
            {'params': conv_params, 'weight_decay': self.config.conv_weight_decay},
            {'params': fc_params, 'weight_decay': self.config.fc_weight_decay},
        ], lr=self.config.lr)


    def trainOneEpoch(self, train_loader, val_loader, epoch,
                      lambda_center, lambda_triplet,
                      c_ratio):

        self.model.train()

        loss_sum_C, loss_sum_P = 0.0, 0.0
        correct, total = 0, 0

        for train_batch, label_batch in train_loader:

            train_batch_C = train_batch[:, self.central_channels, :]
            train_batch_P = train_batch[:, self.parietal_channels, :]

            self.optimizer_C.zero_grad()
            self.optimizer4center_C.zero_grad()

            for name, module in self.model.named_modules():
                if 'branchP' in name or 'conv_P' in name:
                    if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                        module.eval()
                if 'branchC' in name or 'conv_C' in name:
                    if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                        module.train()

            output_batch_C, feature_C, _, _ = self.model(train_batch_C, train_batch_P)
            loss_i_C = self.criterion(output_batch_C, label_batch)
            closs_C = self.centerloss_C(label_batch, feature_C)
            triplet_loss_C = self.triplet_criterion(label_batch, feature_C)
            loss_C = loss_i_C + lambda_center * closs_C + lambda_triplet * triplet_loss_C
            loss_C.backward()
            self.optimizer_C.step()
            self.optimizer4center_C.step()
            loss_sum_C += loss_C.item()

            self.optimizer_P.zero_grad()
            self.optimizer4center_P.zero_grad()

            for name, module in self.model.named_modules():
                if 'branchP' in name or 'conv_P' in name:
                    if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                        module.train()
                if 'branchC' in name or 'conv_C' in name:
                    if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                        module.eval()

            _, _, output_batch_P, feature_P = self.model(train_batch_C, train_batch_P)
            loss_i_P = self.criterion(output_batch_P, label_batch)
            closs_P = self.centerloss_P(label_batch, feature_P)
            triplet_loss_P = self.triplet_criterion(label_batch, feature_P)
            loss_P = loss_i_P + lambda_center * closs_P + lambda_triplet * triplet_loss_P
            loss_P.backward()
            self.optimizer_P.step()
            self.optimizer4center_P.step()
            loss_sum_P += loss_P.item()

            pred = self.gated_decision(output_batch_C, output_batch_P, tau=0.1, c_ratio=c_ratio)

            correct += pred.eq(label_batch).cpu().sum().item()
            total += len(train_batch)

        loss_train_C = round(loss_sum_C / len(train_loader), 6)
        loss_train_P = round(loss_sum_P / len(train_loader), 6)
        acc_train = round(correct / total, 4)

        loss_val_C, loss_val_P, acc_val = self.validate(val_loader)

        return loss_train_C, loss_train_P, acc_train, loss_val_C, loss_val_P, acc_val

    @torch.no_grad()
    def gated_decision(self, out_C, out_P, tau=0.10, c_ratio=0.5):
        pC = F.softmax(out_C, dim=1)
        pP = F.softmax(out_P, dim=1)

        cC, yC = pC.max(dim=1)  # [B]
        cP, yP = pP.max(dim=1)

        # fallback: 融合（推荐log域融合，稳定且符合“乘积/证据累积”）
        fused = (c_ratio * F.log_softmax(out_C, dim=1) + (1 - c_ratio) * F.log_softmax(out_P, dim=1)).argmax(dim=1)

        useC = (cC - cP) > tau
        useP = (cP - cC) > tau
        pred = torch.where(useC, yC, torch.where(useP, yP, fused))
        return pred


    def train(self, lambda_center=0.005, lambda_triplet=0.5,
              c_ratio=0.5, seed=1, split='train_test'):
        self.setRandom(seed)

        all_test_accs = []
        all_test_kappas = []

        all_test_accs_C = []
        all_test_kappas_C = []

        all_test_accs_P = []
        all_test_kappas_P = []

        train_time = 0.0
        inference_time = 0.0

        if self.config.debug:
            exp_subs = [1, 2, 3, 4]
        else:
            exp_subs = np.arange(1, self.data_conf['n_subs'] + 1)

        n_subs = len(exp_subs)
        for sub_id in exp_subs:
            best_vloss = 1e8
            best_vacc = 0.0
            history = {'train_loss_C': [], 'train_loss_P': [],
                       'test_loss_C': [],  'test_loss_P': [],
                       'train_acc': [], 'test_acc': []}
            best_epoch = -1
            lr_drop_epochs = []
            pre_lr = self.config.lr
            X_train, X_test, y_train, y_test = self.load_data(sub_id)
            if split == 'train_val_test':
                X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                              y_train,
                                                              test_size=0.2,
                                                              random_state=seed)
                train_loader = self.gen_loader(X_train, y_train)
                val_loader = self.gen_loader(X_val, y_val)
            else:
                train_loader = self.gen_loader(X_train, y_train)
                val_loader = self.gen_loader(X_test, y_test)

            model_path = f"{self.result_path}/saved_models"
            os.makedirs(model_path, exist_ok=True)
            model_path = f"{model_path}/sub-{sub_id}.pt"

            self.init()

            in_train_time = time.time()
            for epoch in range(self.config.epochs):
                loss_train_C, loss_train_P, acc_train, loss_val_C, loss_val_P, acc_val = self.trainOneEpoch(train_loader, val_loader, epoch,
                                                                              lambda_center=lambda_center, lambda_triplet=lambda_triplet,
                                                                              c_ratio=c_ratio)


                if (
                        self.config.apply_early_stopping and
                        epoch > self.config.early_stopping_patience and
                        acc_val < min(history['test_acc'][-self.config.early_stopping_patience:])
                ):
                    break

                history['train_loss_C'].append(loss_train_C)
                history['train_loss_P'].append(loss_train_P)
                history['train_acc'].append(acc_train)
                history['test_loss_C'].append(loss_val_C)
                history['test_loss_P'].append(loss_val_P)
                history['test_acc'].append(acc_val)
                if acc_val > best_vacc:
                    torch.save({'state_dict': self.model.state_dict()}, model_path)
                    best_vacc = acc_val
                    best_epoch = epoch + 1

                info = (f'Sub-{sub_id}, Epoch [{epoch + 1}/{self.config.epochs}], '
                            f'TLoss_C: {loss_train_C}, TLoss_P: {loss_train_P}, TAcc: {acc_train}, '
                            f'VLoss_C: {loss_val_C}, VLoss_P: {loss_val_P}, VAcc: {acc_val}')

                self.log_writer.write(info + '\n')

                print(info)


            out_train_time = time.time()
            train_time += out_train_time - in_train_time

            history_dir = f"{self.result_path}/history"
            os.makedirs(history_dir, exist_ok=True)
            history_path = f"{history_dir}/history_{self.config.model_name}_sub-{sub_id}.npy"
            np.save(history_path, history)

            if self.config.drawing_learning_curves:
                curve_path = f"{self.result_path}/curves"
                os.makedirs(curve_path, exist_ok=True)
                self.draw_learning_curves(history, sub_id, curve_path, best_epoch, lr_drop_epochs)

            in_inference_time = time.time()
            acc_test, kappa_test, acc_test_C, kappa_test_C, acc_test_P, kappa_test_P = self.predict(X_test, y_test, model_path, c_ratio=c_ratio)
            out_inference_time = time.time()

            all_test_accs.append(acc_test)
            all_test_kappas.append(kappa_test)

            all_test_accs_C.append(acc_test_C)
            all_test_kappas_C.append(kappa_test_C)

            all_test_accs_P.append(acc_test_P)
            all_test_kappas_P.append(kappa_test_P)

            inference_time += out_inference_time - in_inference_time

        train_time = round(train_time / n_subs / 60, 2)
        inference_time = round(inference_time / n_subs, 2)

        avg_acc = round(np.mean(all_test_accs), 4)
        avg_kappa = round(np.mean(all_test_kappas), 4)

        avg_acc_C = round(np.mean(all_test_accs_C), 4)
        avg_kappa_C = round(np.mean(all_test_kappas_C), 4)

        avg_acc_P = round(np.mean(all_test_accs_P), 4)
        avg_kappa_P = round(np.mean(all_test_kappas_P), 4)

        info = f"\n\n------------------{self.exp_id}---------------------------\n"
        info += 'Test performance (%)\n'
        info += '       ' + '    '.join([f"sub_{e}" for e in exp_subs]) + '\n'
        info += '       ' + '    '.join(['----' for i in range(n_subs)]) + '\n'
        info += 'acc:   ' + '    '.join(f"{e * 100:.2f}" for e in all_test_accs) + '\n'
        info += 'acc_C: ' + '    '.join(f"{e * 100:.2f}" for e in all_test_accs_C) + '\n'
        info += 'acc_P: ' + '    '.join(f"{e * 100:.2f}" for e in all_test_accs_P) + '\n'
        info += 'kappa: ' + '    '.join(f"{e * 100:.2f}" for e in all_test_kappas) + '\n'
        info += 'kap_C: ' + '    '.join(f"{e * 100:.2f}" for e in all_test_kappas_C) + '\n'
        info += 'kap_P: ' + '    '.join(f"{e * 100:.2f}" for e in all_test_kappas_P) + '\n'
        info += f'average acc(%)/kappa: {avg_acc * 100:.2f} / {avg_kappa}\n'
        info += f'average acc(%)/kappa-C: {avg_acc_C * 100:.2f} / {avg_kappa_C}\n'
        info += f'average acc(%)/kappa-P: {avg_acc_P * 100:.2f} / {avg_kappa_P}\n'
        info += f'Train time: {train_time}\n'
        info += f'Inference time: {inference_time}\n'

        info += '--------------------------------------------------\n\n'
        print(info)

        self.log_writer.write(info + '\n')
        self.log_writer.close()
        return all_test_accs, all_test_kappas

    def validate(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0.0
            correct, total = 0, 0

            for val_batch, label_batch in val_loader:
                val_batch_C=val_batch[:,self.central_channels,:]
                val_batch_P=val_batch[:,self.parietal_channels,:]

                output_batch_C, _, output_batch_P, _ = self.model(val_batch_C, val_batch_P)
                pred = self.gated_decision(output_batch_C, output_batch_P)

                correct += pred.eq(label_batch).sum().cpu().item()
                total += len(val_batch)
                loss_i_C = self.criterion(output_batch_C, label_batch).item()
                loss_i_P = self.criterion(output_batch_P, label_batch).item()

            return round(loss_i_C / len(val_loader), 6), round(loss_i_P / len(val_loader), 6), round(correct / total, 4)

    def predict(self, X_test, y_test, model_path, c_ratio=0.5):
        with torch.no_grad():

            X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            y_test = torch.tensor(y_test, dtype=torch.long).to(self.device)

            checkpoint = torch.load(model_path)

            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()

            X_test_C=X_test[:,self.central_channels,:]
            X_test_P=X_test[:,self.parietal_channels,:]

            pred_test_C, _, pred_test_P, _ = self.model(X_test_C,X_test_P)

            pred_test = self.gated_decision(pred_test_C, pred_test_P, tau=0.1, c_ratio=c_ratio).cpu().numpy()

            acc_test = round(accuracy_score(y_test.cpu().numpy(), pred_test), 4)
            kappa_test = round(cohen_kappa_score(y_test.cpu().numpy(), pred_test), 4)

            pred_test_C = pred_test_C.argmax(dim=1).cpu().numpy()
            acc_test_C = round(accuracy_score(y_test.cpu().numpy(), pred_test_C), 4)
            kappa_test_C = round(cohen_kappa_score(y_test.cpu().numpy(), pred_test_C), 4)

            pred_test_P = pred_test_P.argmax(dim=1).cpu().numpy()
            acc_test_P = round(accuracy_score(y_test.cpu().numpy(), pred_test_P), 4)
            kappa_test_P = round(cohen_kappa_score(y_test.cpu().numpy(), pred_test_P), 4)

            return acc_test, kappa_test, acc_test_C, kappa_test_C, acc_test_P, kappa_test_P

    def load_data(self, sub_id):
        X_train, X_test, y_train, y_test = get_data(data_path=self.data_conf['data_path'],
                                                    dataset=self.config.data_name,
                                                    subject=sub_id,
                                                    LOSO=False,
                                                    is_standardized=True,
                                                    is_shuffle=True,
                                                    st=2,
                                                    end=4
                                                    )
        return X_train, X_test, y_train, y_test

    def gen_loader(self, x_data, y_data):
        x_data = torch.tensor(x_data, dtype=torch.float32).to(self.device)
        y_data = torch.tensor(y_data, dtype=torch.long).to(self.device)
        return DataLoader(dataset=TensorDataset(x_data, y_data), batch_size=self.config.batch_size, shuffle=True)

    def setRandom(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def draw_learning_curves(self, history, sub, curve_path, best_epoch, lr_drop_epochs):
        plt.figure(figsize=(10, 8))
        plt.plot(history['train_loss'], label='train_loss')
        plt.plot(history['test_loss'], label='test_loss')
        plt.ylim([0, min(10, max(max(history['test_loss']), max(history['train_loss'])))])

        plt.axvline(x=best_epoch, linestyle='--', color='red', linewidth=1)
        best_loss = min(history['test_loss'])
        y_pos = (plt.ylim()[0] + plt.ylim()[1]) * 0.5
        y_text_pos = (plt.ylim()[0] + plt.ylim()[1]) * 0.6
        plt.annotate(f"best_epoch:{best_epoch}\nloss:{best_loss:.4f}",
                     xy=(best_epoch, y_pos),
                     xytext=(best_epoch + 2, y_text_pos),
                     arrowprops=dict(arrowstyle="->", color='red'),
                     color='red',
                     fontsize='16')
        if len(lr_drop_epochs) != 0:
            for lr_drop_epoch in lr_drop_epochs:
                plt.axvline(x=lr_drop_epoch, linestyle='--', color='green', linewidth=1)
                plt.text(x=lr_drop_epoch + 2, y=plt.ylim()[0] + 0.05, s="drop", color='green')

        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.title(f'Learning Curve of {self.config.model_name} in Sub-{sub}')
        plt.tight_layout()
        plt.savefig(f'{curve_path}/loss_sub-{sub}.png')
        plt.show()
        plt.close()
        plt.figure(figsize=(10, 8))
        plt.plot(history['train_acc'])
        plt.plot(history['test_acc'])
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train_acc', 'test_acc'], loc='upper left')
        plt.title(f'Learning Curve of {self.config.model_name} in Sub-{sub}')
        plt.tight_layout()
        plt.savefig(f'{curve_path}/acc_sub-{sub}.png')
        plt.close()


if __name__ == '__main__':
    config = Config(model_name='CrossDA_MNet',
                    data_name='BCI2a',
                    conv_weight_decay=0.3,
                    fc_weight_decay=0.3,
                    apply_early_stopping=False,
                    early_stopping_patience=100,
                    drawing_learning_curves=False,
                    epochs=500,
                    lr=0.001,
                    batch_size=64,
                    )

    trainer = Trainer(config)

    #%% predict
    n_subs = trainer.data_conf['n_subs']
    trainer.init()
    all_accs = []
    all_kappas = []
    for sub_id in range(1, n_subs+1):
        model_path = f'checkpoints/{config.data_name}/sub-{sub_id}.pt'
        _, X_test, _, y_test = trainer.load_data(sub_id)
        acc_test, kappa_test, _, _, _, _ = trainer.predict(X_test, y_test, model_path)
        all_accs.append(acc_test)
        all_kappas.append(kappa_test)
        print(f'sub-{sub_id}, acc, {acc_test}, kappa, {kappa_test}')

    print(f'avg acc, {sum(all_accs)/n_subs:.4f}, avg kappa, {sum(all_kappas)/n_subs:.4f}')


    #%% train (using train_test split) 
    # accs, kappas = trainer.train(lambda_center=0.005, lambda_triplet=0.5, c_ratio=0.5, seed=1, split='train_test')

    #%% train (using train_val_test split) 
    # accs, kappas = trainer.train(lambda_center=0.005, lambda_triplet=0.5, c_ratio=0.5, seed=1, split='train_val_test')

    











