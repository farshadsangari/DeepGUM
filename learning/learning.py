import numpy as np
from tqdm import tqdm
import torch
from torch import optim
import loss 




def SGD_Training(model, cacd_trainLoader, cacd_valLoader, epochs, lr, weight_decay, momentum, loss_threshold, r_train, r_val, batch_size):
    global r,output,y,counter
    break_condition = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), nesterov=True, lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, verbose=False)
    epochs_loss_mse = {"train_loss": [], "val_loss": []}
    epochs_loss_mae = {"train_loss": [], "val_loss": []}
    val_mae_batch_loss = []
    
    for e in range(epochs):
        
        trainloss_mse = 0
        valloss_mse = 0
        trainloss_mae = 0
        valloss_mae = 0
        ##########################            Training Step             #############################################
        model.train()
        print("EPOCH {}/{} :".format(e +1, epochs))
        with tqdm(enumerate(cacd_trainLoader, 1), total=len(cacd_trainLoader), desc ="  train") as t_data_train:
            for counter, (image, age, _) in t_data_train:
                
                r = r_train[(counter-1)*batch_size:counter*batch_size]
                
                r = torch.tensor(r,device=device).clone().detach()
                r.requires_grad = False
                image, age = image.to(device), age.to(device)
                output = model(image).ravel()
#                 MAD = torch.median(torch.abs(output-torch.median(output)))
                loss_mse = loss.weighted_loss(output*torch.sqrt(r), age*torch.sqrt(r), kind='mse')
                loss_mae = loss.weighted_loss(output*torch.sqrt(r), age*torch.sqrt(r), kind='mae')
                optimizer.zero_grad()
                loss_mse.backward()
                optimizer.step()
                trainloss_mse += loss_mse.item()
                trainloss_mae += loss_mae.item()
                batchloss_mse_train = trainloss_mse/counter
                batchloss_mae_train = trainloss_mae/counter
                t_data_train.set_postfix(train_L2=batchloss_mse_train, train_L1=batchloss_mae_train, 
                                         lr=optimizer.state_dict()['param_groups'][0]['lr'], refresh=True)
                
                lr_schedular.step(batchloss_mse_train)

       ##################################           Validation Step           #############################################
        with torch.no_grad():
            model.eval()
            with tqdm(enumerate(cacd_valLoader, 1), total=len(cacd_valLoader), desc ="    val") as t_data_eval:
                for counter, (x, y, _) in t_data_eval:
                    r = r_val[(counter-1)*batch_size:counter*batch_size]
                    r = torch.tensor(r,device=device).clone().detach()
                    r.requires_grad = False
                    x, y = x.to(device), y.to(device)
                    output = model(x).ravel()
                    loss_mse = loss.weighted_loss(output*torch.sqrt(r), y*torch.sqrt(r), kind='mse')
                    loss_mae = loss.weighted_loss(output*torch.sqrt(r), y*torch.sqrt(r), kind='mae')
                    valloss_mse += loss_mse.item()
                    valloss_mae += loss_mae.item()
                    batchloss_mse_val = valloss_mse/counter
                    batchloss_mae_val = valloss_mae/counter
                    t_data_eval.set_postfix(val_L2=batchloss_mse_val, val_L1=batchloss_mae_val, refresh=True)

                    if np.abs(batchloss_mae_val - np.mean(val_mae_batch_loss[-20:])) < loss_threshold:
                        break_condition = True
                        print('- threshold reached -')
                        break
                        
                    val_mae_batch_loss.append(batchloss_mae_val)

                    
                    
        epochs_loss_mse["train_loss"].append(batchloss_mse_train)
        epochs_loss_mse["val_loss"].append(batchloss_mse_val)
        epochs_loss_mae["train_loss"].append(batchloss_mae_train)
        epochs_loss_mae["val_loss"].append(batchloss_mae_val)
        if break_condition == True:
            break
    
    ######################             Compute Final Prediction for EM Step                ################################
    
    torch.cuda.empty_cache()
    y_trues_train = []
    y_preds_train = []
    y_trues_val = []
    y_preds_val = []
    print("Prediction :")
    with tqdm(enumerate(cacd_trainLoader), total=len(cacd_trainLoader) ,desc ="  train") as t_data_train:
        for n, (x, y, _) in t_data_train:
            r = r_train[n*batch_size:(n+1)*batch_size]
            output = model(x.to(device))
            y_trues_train = y_trues_train + [o.item() for o in y]
            y_preds_train = y_preds_train + [o.item() for o in output]
            
    with tqdm(enumerate(cacd_valLoader), total=len(cacd_valLoader) ,desc ="    val") as t_data_eval:
        for n, (x, y, _) in t_data_eval:
            r = r_val[n*batch_size:(n+1)*batch_size]
            output = model(x.to(device))
            y_trues_val = y_trues_val + [o.item() for o in y]
            y_preds_val = y_preds_val + [o.item() for o in output]
        

    return model, epochs_loss_mse, epochs_loss_mae, np.array(y_trues_train).reshape(1,-1), np.array(y_preds_train).reshape(1,-1), np.array(y_trues_val).reshape(1,-1), np.array(y_preds_val).reshape(1,-1)

