
import time
from utils import *
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, TensorDataset,DataLoader
######################################################################################################################### def train
def batching_helper(loader,optimizer,model,use_optimizer,device):
    total_loss = 0
    total_acc = 0
    counter=0
    model.train()
    model.to(device)
    for x_t,y in loader:
        x_t = x_t.to(device)
        y = y.to(device)
        if use_optimizer:
            optimizer.zero_grad()
        out = model(x_t)
        loss = F.nll_loss(out, y)
        acc = accuracy(out, y)
        total_loss+=loss
        total_acc+=acc
        counter+=1
        if use_optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return total_loss/counter,total_acc/counter
def train(model, optimizer, num_epoch, batch_size, data, device, in_dim, embedding_name, mask_type="perClass", keepResult=True, verbose=True, iter_num=None):

    # perform training phase on the input data
    # mdl: model
    # opt: optimizer
    # num_epoch: number of epoch
    # data: input semi-supervised graph
    deltaResults = {}
    increment_len = 5
    increment_counter = 0 
    iter_index = 5
    if mask_type == "allClasses":
        data.train_mask_final = data.trainMask_allClasses
        data.val_mask_final = data.valMask_allClasses
        data.test_mask_final = data.testMask_allClasses
    elif mask_type == "original":
        data.train_mask_final = data.train_mask
        data.val_mask_final = data.val_mask
        data.test_mask_final = data.test_mask
    elif mask_type == "perClass":
        data.train_mask_final = data.trainMask_perClass
        data.val_mask_final = data.valMask_perClass
        data.test_mask_final = data.testMask_perClass

    if keepResult: detailDF = epochPerformanceDF()

    training_mask = data.train_mask_final
    validation_mask = data.val_mask_final
    testing_mask = data.test_mask_final
    y_training = data.y[training_mask]
    y_validation = data.y[validation_mask]
    y_testing = data.y[testing_mask]
    x = data.embedding_vectors[:, 0:in_dim]
    x_training = x[training_mask]
    x_validation = x[validation_mask]
    x_testing = x[testing_mask]
    t=time.time()
    loader_train = DataLoader(TensorDataset(x_training,y_training),shuffle=True,batch_size=batch_size)
    loader_valid = DataLoader(list(zip(x_validation,y_validation)),shuffle=True,batch_size=batch_size)
    loader_test = DataLoader(list(zip(x_testing,y_testing)),shuffle=True,batch_size=batch_size)

    for epoch in range(num_epoch):

        loss_train,acc_train = batching_helper(loader=loader_train,optimizer=optimizer,model=model,use_optimizer=True,device=device)

        with torch.no_grad():
            model.eval()
            loss_val,acc_val = batching_helper(loader=loader_valid,optimizer=optimizer,model=model,use_optimizer=False,device=device)

            loss_test,acc_test = batching_helper(loader=loader_test,optimizer=optimizer,model=model,use_optimizer=False,device=device)

        increment_counter+=1
        if iter_index not in deltaResults:
            deltaResults[iter_index] = (acc_val, acc_test)
        elif deltaResults[iter_index][0] < acc_val:
            deltaResults[iter_index] = (acc_val, acc_test)
        if increment_counter == increment_len:
            increment_counter = 0
            iter_index+=increment_len
        if keepResult:
            dic = {"embedding": embedding_name,"model": model.model_name, "data": data.graph_name, "mask": mask_type,
                "epoch_number": epoch+1, "epoch_time":time.time() - t,
                "training_loss": loss_train.cpu().detach().numpy(), "training_accuracy": acc_train,
                "validation_loss":loss_val.cpu().detach().numpy(), "validation_accuracy": acc_val,
                "test_loss":loss_test.cpu().detach().numpy(), "test_accuracy":acc_test, "iteration_number":iter_num}

            detailDF = appendDF(detailDF, dic)

        if verbose:
            print(detailDF.loc[:, ["embedding", "data", "epoch_number", "epoch_time", "training_loss", "validation_loss", "test_loss",
                                "training_accuracy", "validation_accuracy", "test_accuracy"]].iloc[-1, :])
        print("\n")
    return model, optimizer, detailDF, deltaResults

######################################################################################################################### def test
def test(model, model_name, data, in_dim, batch_size, device, mask_type="manualMask", keepResult=True, verbose=True, iter_num=None):

    sumDF = TrainValidationTestDF()

    if mask_type == "manualMask":
        data.train_mask_final = data.trainMask
        data.val_mask_final = data.valMask
        data.test_mask_final = data.testMask
    else:
        data.train_mask_final = data.train_mask
        data.val_mask_final = data.val_mask
        data.test_mask_final = data.test_mask

    training_mask = data.train_mask_final
    validation_mask = data.val_mask_final
    testing_mask = data.test_mask_final
    y = data.y
    y_training = data.y[training_mask]
    y_validation = data.y[validation_mask]
    y_testing = data.y[testing_mask]
    x = data.embedding_vectors[:, 0:in_dim]
    x_training = x[training_mask]
    x_validation = x[validation_mask]
    x_testing = x[testing_mask]

    x_whole = x
    y_whole = y

    whole_loader = DataLoader(list(zip(x_whole,y_whole)),shuffle=True,batch_size=batch_size)
    loader_train = DataLoader(list(zip(x_training,y_training)),shuffle=True,batch_size=batch_size)
    loader_valid = DataLoader(list(zip(x_validation,y_validation)),shuffle=True,batch_size=batch_size)
    loader_test = DataLoader(list(zip(x_testing,y_testing)),shuffle=True,batch_size=batch_size)


    acc_total = 0
    model.eval()
    model.to(device)
    for x_t,y_t in whole_loader:
        x_t = x_t.to(device)
        y_t = y_t.to(device)
        pred = model(x_t).argmax(dim=1)
        acc_total += (pred==y_t).sum()
        
    acc_model = acc_total/len(y)

    loss_train,acc_train = batching_helper(loader=loader_train,optimizer=None,model=model,use_optimizer=False,device=device)

    loss_val,acc_val = batching_helper(loader=loader_valid,optimizer=None,model=model,use_optimizer=False,device=device)

    loss_test,acc_test = batching_helper(loader=loader_test,optimizer=None,model=model,use_optimizer=False,device=device)

    if keepResult:
        dic = {"model": model_name, "data": data.graph_name, "mask": mask_type, "model_accuracy": acc_model.item(),
               "training_loss": loss_train.cpu().detach().numpy(), "training_accuracy": acc_train,
               "validation_loss": loss_val.cpu().detach().numpy(), "validation_accuracy": acc_val,
               "test_loss": loss_test.cpu().detach().numpy(), "test_accuracy": acc_test, "iteration_number":iter_num}

        sumDF = appendDF(sumDF, dic)

    if verbose:
        print(sumDF.iloc[-1, :])
        print("\n")

    return model, sumDF