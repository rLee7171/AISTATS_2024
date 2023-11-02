import json
import statistics
import matplotlib.pyplot as plt
import numpy as np
root_dir = "/home/ryanlee/Repositories/AISTATS_2024"
dataset_name = ["Cora","CiteSeer","PubMed","WikiCs","Arxiv","Products"]
trun_names = ["truncated_model_0","truncated_model_1","truncated_model_2","truncated_model_3","truncated_model_4"]
conic_names = ["model_0","model_1","model_2","model_3","model_4"]

embedding_list = ["truncated-spectral"]
coeff_list = ["2"]
number_epochs=200
epochs = [str(i) for i in range(1,number_epochs+1) if i%5==0]
condensed_epochs = [str(i) for i in range(1,number_epochs+1) if i%20==0]
num_experiments=20
for dataset in dataset_name:
    for conic_name in conic_names:
        for coeff in coeff_list:
            for embed in embedding_list:
                validation_acc_records = []
                test_acc_records = []
                for trun_name in trun_names:
                    avg_valid_acc = []
                    avg_test_acc = []
                    for experiment_number in range(num_experiments):
                        with open(f"{root_dir}"+"/Truncated_Results/"+f"highest_Validation_{dataset}_conicMLP_{conic_name}_{trun_name}.txt") as jsonFile:
                            model_validation_acc = []
                            model_test_acc = []
                            data = json.load(jsonFile)
                            for e in epochs:
                                model_validation_acc.append(data[coeff][embed][str(experiment_number+1)][e][0])
                                model_test_acc.append(data[coeff][embed][str(experiment_number+1)][e][1])
                            avg_valid_acc.append(model_validation_acc)
                            avg_test_acc.append(model_test_acc)
                    avg_valid_acc = [(sum(i)/num_experiments) for i in zip(*avg_valid_acc)]
                    avg_test_acc = [(sum(i)/num_experiments) for i in zip(*avg_test_acc)]
                    validation_acc_records.append(avg_valid_acc)
                    test_acc_records.append(avg_test_acc)

                record_names = ["Validation","Test"]      
                all_records = [validation_acc_records,test_acc_records]
                for record_index in range(len(all_records)):
                    for index in range(len(all_records[record_index])):
                        plt.plot(epochs,all_records[record_index][index],label=trun_names[index],linestyle="--")

                    plt.xticks(condensed_epochs,rotation="vertical")
                    plt.xlabel("Epochs")
                    plt.ylabel(record_names[record_index]+" Accuracy")
                    plt.title(f"Dataset: {dataset}, Embed: {embed}")
                    plt.legend()
                    plt.subplots_adjust(bottom=0.2)
                    plt.savefig(f"{root_dir}/Truncated_Graphs/{dataset}_Conic_{conic_name}_{record_names[record_index]}_coeff_{coeff}_embed_{embed}.png")
                    plt.close()
                
                



