import os

import torch.cuda
from tensorboardX import SummaryWriter
from henry.mnist_vae_pure import *
from tqdm import tqdm
from WeightsModification import *

from src.utils import *

LATENT_DIMS = 2

def federate(args, custom_client_weights=None, custom_client_datasets=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    torch.cuda.set_device('cuda:0')
    device = 'cuda'

    train_dataset, test_dataset, user_groups = get_dataset(args)
    idxs_users = np.array(list(range(args.num_users)))
    client_datasets = [ClientDatasetManager(train_dataset, user_groups[i]) for i in
                       idxs_users] if custom_client_datasets is None else custom_client_datasets
    client_losses = [ClientLossManager() for _ in idxs_users]
    client_weights = calculate_relative_dataset_sizes(
        client_datasets) if custom_client_weights is None else custom_client_weights

    global_model = construct_model(args)
    global_weights = global_model.state_dict()

    global_loss_manager = ClientLossManager()

    for epoch in tqdm(range(args.epochs)):
        local_weights = []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        for user_idx, client_dataset_manager in enumerate(client_datasets):
            # Train client for this epoch
            print(f"Training user {user_idx} in round {epoch + 1}")
            local_model = construct_model(args)
            local_model.load_state_dict(copy.deepcopy(global_weights))
            local_model.train()
            li_total, li_mse, li_kl = local_model.train_model(client_dataset_manager.training_subset, args.local_bs,
                                                              args.local_ep, beta=args.beta)
            client_losses[user_idx].add_training_losses(li_total, li_mse, li_kl)
            local_weights.append(copy.deepcopy(local_model.state_dict()))

            # local_model.eval()
            # total_val_loss, mse_val_loss, kl_val_loss = local_model.evaluate_model(test_dataset,batch_size=1, beta=args.beta)
            # print(
            #     f"(Test Set) user {user_idx} in round {epoch + 1} totalL: {total_val_loss} mseL: {mse_val_loss} klL: {kl_val_loss}")
            #
            # client_losses[user_idx].add_validation_losses(total_val_loss, mse_val_loss, kl_val_loss)

        global_weights = fed_avg(local_weights, client_weights)
        global_model.load_state_dict(global_weights)
        global_model.eval()
        total_test_loss, mse_test_loss, kl_test_loss = global_model.evaluate_model(test_dataset, 1)
        global_loss_manager.add_validation_losses(total_test_loss, mse_test_loss, kl_test_loss)
        print(f"TEST LOSS AT GLOBAL ROUND {epoch + 1} totalL: {total_test_loss}")


    print("TRAINING ALL DONE!")
    return FederationResult(global_model, client_losses, client_datasets, global_loss_manager)


def construct_model(args):
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        return VariationalAutoencoder(latent_dims=LATENT_DIMS)
    elif args.dataset == 'cifar':
        raise NotImplementedError
    else:
        raise ValueError
