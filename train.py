import numpy as np
import torch
import models
import learning
import util
from data import dataloaders


def _main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.VGG_LinearRegression()
    model.to(device)

    train_loader, test_loader, val_loader = dataloaders(
        args.all_dirs_path,
        args.num_train_data,
        args.num_test_data,
        args.num_val_data,
        args.batch_size,
        args.shuffle,
        args.num_workers,
    )

    #############################    Initialize rn_s with ONE    ############################################
    rn_s_train = np.ones(args.num_train_data)
    rn_s_val = np.ones(args.num_val_data)

    ##################################    Run SGD Algorithm      ###########################################

    (
        model,
        epochs_loss_mse,
        epochs_loss_mae,
        y_trues_train,
        y_preds_train,
        y_trues_val,
        y_preds_val,
    ) = learning.SGD_Training(
        model=model,
        cacd_trainLoader=train_loader,
        cacd_valLoader=val_loader,
        epochs=args.max_epochs_vgg1,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        loss_threshold=args.loss_threshold,
        r_train=rn_s_train,
        r_val=rn_s_val,
        batch_size=args.batch_size,
    )

    #######################################      EM & SGD      ###########################################
    EM_statistics = {"Rs": [], "MSE_loss": [], "MAE_loss": []}
    EM_statistics["MSE_loss"].append(epochs_loss_mse)
    EM_statistics["MAE_loss"].append(epochs_loss_mae)

    cov = args.cov
    prior = args.prior
    gamma = args.gamma

    for _ in range(args.max_epochs_em_sgd):

        ###############   EM Algorithm    #############
        cov, prior, gamma, rn_s_train, rn_s_val, Rs = models.EM_algorithm(
            y_preds_train,
            y_trues_train,
            y_preds_val,
            y_trues_val,
            cov=cov,
            prior=prior,
            gamma=gamma,
            max_iter_em=args.max_epochs_em,
            param_threshold=[1e-8, 1e-8, 1e-8],
        )

        ##############   SGD    ####################
        (
            model,
            epochs_loss_mse,
            epochs_loss_mae,
            y_trues_train,
            y_preds_train,
            y_trues_val,
            y_preds_val,
        ) = learning.SGD_Training(
            model,
            train_loader,
            test_loader,
            epochs=args.max_epochs_vgg2,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            loss_threshold=args.loss_threshold,
            r_train=rn_s_train,
            r_val=rn_s_val,
            batch_size=args.batch_size,
        )
        EM_statistics["Rs"].append(Rs)
        EM_statistics["MSE_loss"].append(epochs_loss_mse)
        EM_statistics["MAE_loss"].append(epochs_loss_mae)

    return model, rn_s_train, rn_s_val, EM_statistics


if __name__ == "__main__":
    args = util.get_args()
    _main(args)
