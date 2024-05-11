
import os

import torch.cuda
from tensorboardX import SummaryWriter
from henry.mnist_vae_pure import *
from tqdm import tqdm
from ClientWrapper import *

from src.utils import *

LATENT_DIMS = 2
LOCAL_EPOCHS = 10
LOCAL_TRAINING_BATCH_SIZE = 16
def federate(args):
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    torch.cuda.set_device('cuda:0')
    device = 'cuda'

    train_dataset, test_dataset, user_groups = get_dataset(args)
    idxs_users = np.array(list(range(args.num_users)))
    client_datasets = [ClientDatasetManager(train_dataset, user_groups[i]) for i in idxs_users]
    client_losses = [ClientLossManager() for _ in idxs_users]


    dataset_size_per_client = [client_dataset.train_length() for client_dataset in client_datasets]

    global_model = None

    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        global_model = VariationalAutoencoder(latent_dims=LATENT_DIMS)

    global_model.to(device)
    global_model.train()

    global_weights = global_model.state_dict()
    for epoch in tqdm(range(args.epochs)):
        local_weights = []
        print(f'\n | Global Training Round : {epoch +1} |\n')
        global_model.train()

        for user_idx, client_dataset_manager in enumerate(client_datasets):
            # Train client for this epoch
            print(f"Training user {user_idx} in round {epoch + 1}")
            local_model = copy.deepcopy(global_model)
            local_model.train()
            li_total, li_mse, li_kl = local_model.train_model(client_dataset_manager.training_subset, LOCAL_TRAINING_BATCH_SIZE, LOCAL_EPOCHS)
            client_losses[user_idx].add_training_losses(li_total, li_mse, li_kl)
            local_weights.append(copy.deepcopy(local_model.state_dict()))

            # Validate client for this epoch
            local_model.eval()
            validation_dataset = client_dataset_manager.validation_subset
            total_val_loss, mse_val_loss, kl_val_loss = local_model.evaluate_model(validation_dataset, int(len(validation_dataset) / 10))
            client_losses[user_idx].add_validation_losses(total_val_loss, mse_val_loss, kl_val_loss)

        global_weights = fed_avg(local_weights, dataset_size_per_client)
        global_model.load_state_dict(global_weights)

    global_model.eval()
    total_test_loss, mse_test_loss, kl_test_loss = global_model.evaluate_model(test_dataset, 128)
    client_losses[user_idx].add_validation_losses(total_test_loss, mse_test_loss, kl_test_loss)

    return global_model, client_losses









