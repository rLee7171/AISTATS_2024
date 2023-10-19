globals().clear()
######################################################################################################################## libraries
import gc
import json
import time
from spectrumMLP import *
from train import *
from data import *
from utils import *
from parameters import *
from truncated_SpectralEmbedding import *

#########################################################################################################################

root_dir = "/home/ryanlee/Repositories/AISTATS_2024"

dataset_name = ["Products"]

######################################################################################################################### 

torch.cuda.empty_cache()
gc.collect()

device = gpu_setup(use_gpu)

if device.type == 'cuda':
  torch.cuda.manual_seed(123)
  torch.cuda.manual_seed_all(123)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


######################################################################################################################### data loading and preperation
# hyper-parameter initialization

learning_rate = 0.01
num_epoch = 200
num_exp = 20                                       # number of experiments
use_cache = True
batch_size=2048
embedding_list = ["non-symmetric", "symmetric","deepwalk"]
coeff_list = [2]
                                                   

epochResults = epochPerformanceDF()  # detailed of each epoch for train and validation set, both accuracy and loss
summaryResults = TrainValidationTestDF()  # summary of trained model for train, validation, and test, both accuracy and loss

##################################################### datasets
for ds in dataset_name:
    #Multi-nested Dictionary: first index denotes coeff, second index denotes embedding, third denotes experiment number, 
    #fourth denotes every "delta" epochs which records the highest valid accuracy with corresponding test accuracy
    highest_validation = {}
    for coeff in coeff_list:
        highest_validation[coeff] = {}
        for embed in embedding_list:
            highest_validation[coeff][embed] = {}
            name = f"coeff={coeff}+embed={embed}"
            print(f"{name} \n")
            
            gc.collect()

            graph = data_prepare(dataset_name=ds, maskInd=maskInd, root_dir=root_dir)

            graph = trainValidationTest_splitPerClass(data=graph, trainVal_percent=trainVal_percent_perClass,
                                                        train_percent=train_percent_perClass,
                                                        train_num=train_num_perClass, val_num=val_num_perClass,
                                                        verbose=data_verbose)

            graph = trainValidationTest_splitAllClasses(graph, train_percent_allClasses, train_num_allClasses,
                                                            val_percent_allClasses, val_num_allClasses,
                                                            verbose=data_verbose)

            graph.dir = root_dir

            if embed in ["non-symmetric", "symmetric"]:
                graph = embedding(data=graph, dataset_name=graph.graph_name, root_dir=graph.dir,
                                    ncol=coeff * graph.num_classes, param=coeff, drp_first=True, mode=embed,use_cache=True)

            elif embed=="deepwalk":
                graph.embedding_vectors = deepwalk(data=graph, root_dir=root_dir, dataset_name=ds, device=device,
                                                    emb_dim=coeff*graph.num_classes, param=coeff, learning_rate=deepwalk_lr,
                                                    n_epoch=deepwalk_epoch, mask_type="original",
                                                    batch_size=deepwalk_batchSize,use_cache=True)
            for iter_num in range(num_exp):

                ##################################################### model: spectrumMLP
                gt = time.time()
                torchStatus()
                torch.cuda.seed_all()
                print("\n")

                ##################################################### initialization

                st = time.time()
                mdl = spectrumMLP(in_dim=coeff*graph.num_classes, hidden_dim=4*graph.num_classes,
                                    out_dim=graph.num_classes, num_lin=num_linear, add_relu=add_relu,
                                    bias=mlp_bias, init=init, pdrop=pdrop, embedding=embed,
                                    deepwalk_epoch=deepwalk_epoch, deepwalk_lr=deepwalk_lr,
                                    deepwalk_maskType=deepwalk_maskType, deepwalk_batchSize=deepwalk_batchSize,
                                    cache=use_cache)

                print(f"model is {mdl.model_name} \n")
                mdl_name = mdl.model_name
                opt = torch.optim.Adam(mdl.parameters(), lr=learning_rate)

                ##################################################### training phase
                print(coeff*graph.num_classes)
                print("entering training phase...\n")
                mdl, opt, epochDF, deltaResults = train(model=mdl, optimizer=opt, mask_type=mask_type,
                                                        num_epoch=num_epoch, batch_size=batch_size, device=device, embedding_name = embed, in_dim=coeff*graph.num_classes, data=graph, keepResult=train_keep,
                                                        verbose=train_verbose, iter_num=iter_num)
                highest_validation[coeff][embed][iter_num+1]=deltaResults
                t_train = time.time() - st

                epochResults = pd.concat([epochResults, epochDF], ignore_index=True)

                print("\n")

                ########################################## end of training phase

                ##################################################### test phase
                print("entering test phase...\n")
                st = time.time()

                mdl, sumDF = test(model=mdl, model_name=mdl_name, data=graph, in_dim=coeff*graph.num_classes, batch_size=batch_size, mask_type=mask_type,
                                    keepResult=test_keep, verbose=test_verbose, iter_num=iter_num,device=device)
                t_test = time.time() - st
                summaryResults = pd.concat([summaryResults, sumDF], ignore_index=True)

                ft = time.time() - gt
                del mdl
                torch.cuda.empty_cache()
                gc.collect()

            del graph
            torch.cuda.empty_cache()
            gc.collect()

    ##########################################################################################  save results
    with open(f"{root_dir}/Results/highest_Validation_{ds}_spectrumMLP.txt", "w") as fp:
        json.dump(highest_validation, fp)

