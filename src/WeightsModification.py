import numpy as np
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
import pickle
import pandas as pd

class ClientDatasetManager:
    def __init__(self, dataset, idxs):
        # np.random.shuffle(idxs)
        # training_idxs, validation_idxs = idxs[:int(0.8 * len(idxs))], idxs[int(0.8 * len(idxs)):]
        # training_idxs = [int(i) for i in training_idxs]
        # validation_idxs= [int(i) for i in validation_idxs]
        self.training_subset = Subset(dataset, idxs)
        # self.validation_subset = Subset(dataset, validation_idxs)

    def __len__(self):
        self.train_length()

    def train_length(self):
        return len(self.training_subset)

    # def val_length(self):
    #     return len(self.validation_subset)

    def _get_dataset_split(self):
        whole_dataset = self.training_subset.dataset
        whole_labels = np.array(whole_dataset.targets)
        relevant_idxs = self.training_subset.indices
        res = [0] * len(np.unique(whole_labels))
        for i in relevant_idxs:
            res[whole_labels[i]] += 1
        return res

    @staticmethod
    def plot_dataset_splits(client_dataset_managers):
        client_splits = [manager._get_dataset_split() for manager in client_dataset_managers]
        columns = ["Client","0","1","2","3","4","5","6","7","8","9"]

        for i in range(len(client_splits)):
            client_splits[i].insert(0, i)

        df = pd.DataFrame(client_splits, columns=columns)
        df.plot(x = 'Client', kind='bar', stacked=False)


        # class_splits = []
        # num_classes = len(np.unique(np.array(client_dataset_managers[0].training_subset.dataset.targets)))
        # class_splits = [[] for _ in range(num_classes)]
        # for client_dataset_manager in client_dataset_managers:
        #     client_split = client_dataset_manager._get_dataset_split()
        #     for i in range(len(client_split)):
        #         class_splits[i].append(client_split[i])
        #
        # # x = [f"Client {client_idx}" for client_idx in range(len(client_dataset_managers))]
        # x = np.arange(len(client_dataset_managers))
        # width = 0.2
        # multiplier = 0
        #
        # fig, ax = plt.subplots(layout='constrained')
        # for class_split in class_splits:
        #     offset = width * multiplier
        #     rects = ax.bar(x + offset, class_split, width)
        #     ax.bar_label(rects, padding=3)
        #     multiplier += 1
        #
        # ax.set_ylabel('Frequency')
        # ax.set_xticks(x + width, [f"Client {client_idx}" for client_idx in x])
        # # ax.set_ylim(0,250)
        # plt.show()











class ClientLossManager:

    def __init__(self):
        self.train_total_across_communication = []
        self.train_mse_across_communication = []
        self.train_kl_across_communication = []
        self.validation_total_across_communication = []
        self.validation_mse_across_communication = []
        self.validation_kl_across_communication = []
        self.test_total_loss = 0
        self.test_mse_loss = 0
        self.test_kl_loss = 0

    def add_training_losses(self, li_total, li_mse, li_kl):
        self.train_total_across_communication.append(np.average(li_total))
        self.train_mse_across_communication.append(np.average(li_mse))
        self.train_kl_across_communication.append(np.average(li_kl))

    def add_validation_losses(self, total, mse, kl):
        self.validation_total_across_communication.append(total)
        self.validation_mse_across_communication.append(mse)
        self.validation_kl_across_communication.append(kl)


    @staticmethod
    def process_data(client_loss_managers):

        train_total_loss = []
        train_mse_loss = []
        train_kl_loss = []

        validation_total_loss = []
        validation_mse_loss = []
        validation_kl_loss = []

        num_comm_rounds = len(client_loss_managers[0].train_total_across_communication)
        for round_idx in range(num_comm_rounds):
            round_train_total_loss = 0
            round_train_mse_loss = 0
            round_train_kl_loss = 0
            round_validation_total_loss = 0
            round_validation_mse_loss = 0
            round_validation_kl_loss = 0
            for client_idx in range(len(client_loss_managers)):
                round_train_total_loss += client_loss_managers[client_idx].train_total_across_communication[round_idx]
                round_train_mse_loss += client_loss_managers[client_idx].train_mse_across_communication[round_idx]
                round_train_kl_loss += client_loss_managers[client_idx].train_kl_across_communication[round_idx]
                round_validation_total_loss += client_loss_managers[client_idx].validation_total_across_communication[round_idx]
                round_validation_mse_loss += client_loss_managers[client_idx].validation_mse_across_communication[round_idx]
                round_validation_kl_loss += client_loss_managers[client_idx].validation_kl_across_communication[round_idx]

            train_total_loss.append(round_train_total_loss / len(client_loss_managers))
            train_mse_loss.append(round_train_mse_loss / len(client_loss_managers))
            train_kl_loss.append(round_train_kl_loss / len(client_loss_managers))
            validation_total_loss.append(round_validation_total_loss / len(client_loss_managers))
            validation_mse_loss.append(round_validation_mse_loss / len(client_loss_managers))
            validation_kl_loss.append(round_validation_kl_loss / len(client_loss_managers))

        return train_total_loss, train_mse_loss, train_kl_loss, validation_total_loss, validation_mse_loss, validation_kl_loss






class FederationResult:
    def __init__(self, global_model, all_losses, client_datasets, global_loss_manager):
        self.global_model = global_model
        self.all_losses = all_losses
        self.client_datasets= client_datasets
        self.global_loss_manager = global_loss_manager

    def serialise(self, identifier, args):
        file_name = '../../save/objects/{}_{}_{}_COMMS{}_LOCAL{}_BS{}_USERS{}.pkl'. \
            format(identifier,args.dataset, "VAE", args.epochs, args.local_ep, args.local_bs, args.num_users)

        with open(file_name, 'wb') as f:
            pickle.dump(self, f)



