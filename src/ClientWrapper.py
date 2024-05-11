import numpy as np
from torch.utils.data import Subset, DataLoader


class ClientDatasetManager:
    def __init__(self, dataset, idxs):
        np.random.shuffle(idxs)
        training_idxs, validation_idxs = idxs[:int(0.8 * len(idxs))], idxs[int(0.8 * len(idxs)):]
        training_idxs = [int(i) for i in training_idxs]
        validation_idxs= [int(i) for i in validation_idxs]
        self.training_subset = Subset(dataset, training_idxs)
        self.validation_subset = Subset(dataset, validation_idxs)

    def __len__(self):
        self.train_length()

    def train_length(self):
        return len(self.training_subset)

    def val_length(self):
        return len(self.validation_subset)


class ClientLossManager:

    def __init__(self):
        self.train_total_across_communication = []
        self.train_mse_across_communication = []
        self.train_kl_across_communication = []
        self.validation_total_across_communication = []
        self.validation_mse_across_communication = []
        self.validation_kl_across_communication = []

    def add_training_losses(self, li_total, li_mse, li_kl):
        self.train_total_across_communication.append(np.average(li_total))
        self.train_mse_across_communication.append(np.average(li_mse))
        self.train_kl_across_communication.append(np.average(li_kl))

    def add_validation_losses(self, total, mse, kl):
        self.validation_total_across_communication.append(total)
        self.validation_mse_across_communication.append(mse)
        self.validation_kl_across_communication.append(kl)
