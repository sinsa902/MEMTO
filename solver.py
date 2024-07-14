# Some code based on https://github.com/thuml/Anomaly-Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.Transformer import TransformerVar
from model.loss_functions import *
from data_factory.data_loader import get_loader_segment
import logging
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import lib as lb

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class TwoEarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name="", delta=0, type=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif (
            score < self.best_score + self.delta
            or score2 < self.best_score2 + self.delta
        ):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(
            model.state_dict(),
            os.path.join(path, str(self.dataset) + "_checkpoint.pth"),
        )
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class OneEarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name="", delta=0, type=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.dataset = dataset_name
        self.type = type

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        torch.save(
            model.state_dict(),
            os.path.join(path, str(self.dataset) + f"_checkpoint_{self.type}.pth"),
        )
        self.val_loss_min = val_loss


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        (
            self.rawdata,
            self.train_loader,
            self.vali_loader,
            self.k_loader,
            self.test_loader,
        ) = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            dataset=self.dataset,
        )

        self.thre_loader = self.vali_loader
        if self.mode == "memory_initial":
            self.memory_initial = True
            self.phase_type = "second_train"
        else:
            self.memory_initial = False
            if self.mode == "train":
                self.phase_type = "train"
            else:
                self.phase_type = "test"

        self.memory_init_embedding = None

        self.build_model(memory_init_embedding=self.memory_init_embedding)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.entropy_loss = EntropyLoss()
        self.criterion = (
            nn.MSELoss()
            if self.dataset != "adult"
            else F.binary_cross_entropy_with_logits
        )

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def build_model(self, memory_init_embedding):
        self.model = TransformerVar(
            d_numerical=0
            if self.rawdata.X_num is None
            else self.rawdata.X_num["train"].shape[1],
            categories=lb.get_categories(self.rawdata.X_cat),
            c_out=self.output_c,
            e_layers=3,
            d_model=self.d_model,
            n_memory=self.n_memory,
            device=self.device,
            memory_initial=self.memory_initial,
            memory_init_embedding=memory_init_embedding,
            phase_type=self.phase_type,
            dataset_name=self.dataset,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def vali(self, vali_loader):
        self.model.eval()

        valid_loss_list = []
        valid_re_loss_list = []
        valid_entropy_loss_list = []

        for i, (x_num, x_cat, labels) in enumerate(vali_loader):
            x_num = x_num.float().to(self.device)
            x_cat = x_cat.to(self.device)
            labels = labels.float().to(self.device)
            output_dict = self.model(x_num, x_cat)
            output, queries, mem_items, attn = (
                output_dict["out"],
                output_dict["queries"],
                output_dict["mem"],
                output_dict["attn"],
            )

            rec_loss = self.criterion(output, labels)
            entropy_loss = self.entropy_loss(attn)
            loss = rec_loss + self.lambd * entropy_loss

            valid_re_loss_list.append(rec_loss.detach().cpu().numpy())
            valid_entropy_loss_list.append(entropy_loss.detach().cpu().numpy())
            valid_loss_list.append(loss.detach().cpu().numpy())

        return (
            np.average(valid_loss_list),
            np.average(valid_re_loss_list),
            np.average(valid_entropy_loss_list),
        )

    def train(self, training_type):
        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = OneEarlyStopping(
            patience=10, verbose=True, dataset_name=self.dataset, type=training_type
        )
        train_steps = len(self.train_loader)

        from tqdm import tqdm

        for epoch in tqdm(range(self.num_epochs)):
            iter_count = 0
            loss_list = []
            rec_loss_list = []
            entropy_loss_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (x_num, x_cat, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                x_num = x_num.float().to(self.device)
                x_cat = x_cat.to(self.device)
                labels = labels.float().to(self.device)
                output_dict = self.model(x_num, x_cat)

                output, memory_item_embedding, queries, mem_items, attn = (
                    output_dict["out"],
                    output_dict["memory_item_embedding"],
                    output_dict["queries"],
                    output_dict["mem"],
                    output_dict["attn"],
                )

                rec_loss = self.criterion(output, labels)
                entropy_loss = self.entropy_loss(attn)
                loss = rec_loss + self.lambd * entropy_loss

                loss_list.append(loss.detach().cpu().numpy())
                entropy_loss_list.append(entropy_loss.detach().cpu().numpy())
                rec_loss_list.append(rec_loss.detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()
                try:
                    loss.mean().backward()

                except:
                    import pdb

                    pdb.set_trace()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(loss_list)
            train_entropy_loss = np.average(entropy_loss_list)
            train_rec_loss = np.average(rec_loss_list)
            valid_loss, valid_re_loss_list, valid_entropy_loss_list = self.vali(
                self.vali_loader
            )

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, valid_loss
                )
            )
            print(
                "Epoch: {0}, Steps: {1} | VALID reconstruction Loss: {3:.7f} Entropy loss Loss: {2:.7f}  ".format(
                    epoch + 1, train_steps, valid_re_loss_list, valid_entropy_loss_list
                )
            )
            print(
                "Epoch: {0}, Steps: {1} | TRAIN reconstruction Loss: {3:.7f} Entropy loss Loss: {2:.7f}  ".format(
                    epoch + 1, train_steps, train_rec_loss, train_entropy_loss
                )
            )

            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return memory_item_embedding

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(
                    str(self.model_save_path),
                    str(self.dataset) + "_checkpoint_second_train.pth",
                )
            )
        )
        self.model.eval()

        print("======================TEST MODE======================")

        criterion = self.criterion
        gathering_loss = GatheringLoss(reduce=False)
        temperature = self.temperature

        reconstructed_output = []
        original_output = []
        rec_loss_list = []

        test_labels = []
        test_attens_energy = []
        for i, (x_num, x_cat, labels) in enumerate(self.test_loader):
            x_num = x_num.float().to(self.device)
            x_cat = x_cat.to(self.device)
            labels = labels.float().to(self.device)
            output_dict = self.model(x_num, x_cat)

            output, queries, mem_items = (
                output_dict["out"],
                output_dict["queries"],
                output_dict["mem"],
            )

            rec_loss = torch.mean(criterion(labels, output), dim=-1)
            latent_score = torch.softmax(
                gathering_loss(queries, mem_items) / temperature, dim=-1
            )
            loss = latent_score * rec_loss

            cri = loss.detach().cpu().numpy()
            test_attens_energy.append(cri)
            test_labels.append(labels.detach().cpu().numpy())
            reconstructed_output.append(output.detach().cpu().numpy())
            rec_loss_list.append(rec_loss.detach().cpu().numpy())

        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_output = np.concatenate(reconstructed_output, axis=0).reshape(-1)

        test_labels = np.array(test_labels)
        test_output = np.array(test_output)

        test_output_tensor = torch.tensor(test_output)
        pred = (torch.sigmoid(test_output_tensor) >= 0.5).int().numpy()
        gt = test_labels.astype(int)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average="binary"
        )
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision, recall, f_score
            )
        )
        print("=" * 50)

        self.logger.info(f"Dataset: {self.dataset}")
        self.logger.info(f"number of items: {self.n_memory}")
        self.logger.info(f"Precision: {round(precision,4)}")
        self.logger.info(f"Recall: {round(recall,4)}")
        self.logger.info(f"f1_score: {round(f_score,4)} \n")
        return accuracy, precision, recall, f_score

    def get_memory_initial_embedding(self, training_type="second_train"):
        self.model.load_state_dict(
            torch.load(
                os.path.join(
                    str(self.model_save_path),
                    str(self.dataset) + "_checkpoint_first_train.pth",
                )
            )
        )
        self.model.eval()

        for i, (x_num, x_cat, labels) in enumerate(self.k_loader):
            x_num = x_num.float().to(self.device)
            x_cat = x_cat.to(self.device)
            if i == 0:
                output = self.model(x_num, x_cat)["queries"]
            else:
                output = torch.cat([output, self.model(x_num, x_cat)["queries"]], dim=0)

        self.memory_init_embedding = k_means_clustering(
            x=output, n_mem=self.n_memory, d_model=self.d_model, device=self.device
        )

        self.memory_initial = False

        self.build_model(memory_init_embedding=self.memory_init_embedding.detach())

        memory_item_embedding = self.train(training_type=training_type)

        memory_item_embedding = memory_item_embedding[: int(self.n_memory), :]

        item_folder_path = "memory_item"
        if not os.path.exists(item_folder_path):
            os.makedirs(item_folder_path)

        item_path = os.path.join(
            item_folder_path, str(self.dataset) + "_memory_item.pth"
        )

        torch.save(memory_item_embedding, item_path)
