globals().clear()
######################################################################################################################## libraries
import gc
import json
from conicMLP import *
from spectrumMLP import *
from train import *
from data import *
from utils import *
from parameters import *
from truncated_SpectralEmbedding import *

######################################################################################################################### 

root_dir = "/home/ryanlee/Repositories/AISTATS_2024"

#########################################################################################################################

dataset_name = ["Cora", "CiteSeer", "PubMed", "WikiCs", "Arxiv", "Products"] # dataset name
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
num_exp = 20                                       # number of experiments
use_cache = True
batch_size = 128
embedding_list = ["non-symmetric", "symmetric","deepwalk","truncated-spectral"]
coeff_list = [2]
iterations = [2,4,8,16,32]
for ds in dataset_name:
    compute_times = {}
    for coeff in coeff_list:
        for embed in embedding_list:
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
                start = time.time()
                graph = embedding(data=graph, dataset_name=graph.graph_name, root_dir=graph.dir,
                                        ncol=coeff*graph.num_classes, param=coeff, drp_first=True, mode=embed,use_cache=False)
                end = time.time()
                compute_times[embed] = (end-start)
            elif embed=="deepwalk":
                start = time.time()
                graph.embedding_vectors = deepwalk(data=graph, root_dir=root_dir, dataset_name=ds, device=device,
                                                        emb_dim=coeff*graph.num_classes, param=coeff, learning_rate=deepwalk_lr,
                                                        n_epoch=deepwalk_epoch, mask_type="original",
                                                        batch_size=deepwalk_batchSize,use_cache=False)
                end = time.time()
                compute_times[embed] = (end-start)
            elif embed=="truncated-spectral":
                compute_times[embed] = {}
                for iter in iterations:
                    start = time.time()
                    eigenVec, lambdaVal, runTime = truncated_spectral_embedding(G=graph, root_dir=root_dir, dataset_name=ds, dim=coeff*graph.num_classes, iter=iter, amb_dim=None,use_cache=False)
                    end = time.time()
                    compute_times[embed][iter] = (end-start)


            del graph
            torch.cuda.empty_cache()
            gc.collect()
    with open(f"{root_dir}/compute_Embedding_Times/compute_times_{ds}.txt", "w") as fp:
            json.dump(compute_times, fp)
