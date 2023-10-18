

globals().clear()
######################################################################################################################## libraries
import gc
import json
from spectrumMLP import *
from train import *
from data import *
from utils import *
from parameters import *
from truncated_SpectralEmbedding import *

######################################################################################################################### hyper-parameter initialization

root_dir = "/home/ryanlee/Repositories/AISTATS_2024"

#########################################################################################################################

dataset_name = ["Cora", "CiteSeer", "PubMed", "WikiCs", "Arxiv"] # dataset name
model_name = ["truncated_model_1","truncated_model_3"]
model_name = ["truncated_model_0","truncated_model_1","truncated_model_2","truncated_model_3", "truncated_model_4"]
model_dic = {"truncated_model_0": 2,"truncated_model_1": 4, "truncated_model_2": 8, "truncated_model_3": 16, "truncated_model_4": 32}
######################################################################################################################### results df

torch.cuda.empty_cache()
gc.collect()

device = gpu_setup(use_gpu)

if device.type == 'cuda':
  torch.cuda.manual_seed(123)
  torch.cuda.manual_seed_all(123)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


######################################################################################################################### data loading and preperation
# graph: a list of prepared graph datasets

learning_rate = 0.01
num_epoch = 200
num_exp = 20                                      # number of experiments
batch_size=128
use_cache = True
embedding_list=["truncated-spectral"]
coeff_list = [2]
weight_decay_list = [0]                            # add other real number is needed

epochResults = epochPerformanceDF()  # detailed of each epoch for train and validation set, both accuracy and loss
summaryResults = TrainValidationTestDF()  # summary of trained model for train, validation, and test, both accuracy and loss

##################################################### datasets
for ds in dataset_name:
    for mdl_n in model_name:
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

                for iter_num in range(num_exp):
                    eigenVec, lambdaVal, runTime = truncated_spectral_embedding(G=graph, root_dir=root_dir, dataset_name=ds, dim=coeff*graph.num_classes, iter=model_dic[mdl_n], amb_dim=None,use_cache=False)
                    graph.embedding_vectors = eigenVec

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
                                                            num_epoch=num_epoch, batch_size=batch_size, device=device, in_dim=coeff*graph.num_classes, embedding_name=mdl_n, data=graph, keepResult=train_keep,
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

                del graph
                torch.cuda.empty_cache()
                gc.collect()
        with open(f"{root_dir}/Truncated_Results/highest_Validation_{ds}_truncated_spectrumMLP_{mdl_n}.txt", "w") as fp:
            json.dump(highest_validation, fp)
