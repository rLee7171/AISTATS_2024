import json
import statistics
import matplotlib.pyplot as plt
import numpy as np
root_dir = "/home/ryanlee/Repositories/AISTATS_2024"

dataset_name = ["Cora","CiteSeer","PubMed","WikiCs","Arxiv","Products"]

model_name = ["spectrumMLP"]

different_model_name = "MLP"

embedding_list = ["non-symmetric","truncated-spectral"]

different_embed_names = ["Spectral$_{NS}$", "Truncated$_{32}$"]

combined_name = "truncated_spec"

coeff_list = ["2"]
number_epochs=200
epochs = [str(i) for i in range(1,number_epochs+1) if i%5==0]
condensed_epochs = [str(i) for i in range(1,number_epochs+1) if i%20==0]
num_experiments=20

for dataset in dataset_name:
    average_performance_dic = {}
    for model in model_name:
        for coeff in coeff_list:
            validation_acc_records = []
            test_acc_records = []
            for embed in embedding_list:
                average_performance_dic[embed] = {}
                average_performance_dic[embed]["avg_valid"] = {}
                average_performance_dic[embed]["avg_test"] = {}
                avg_valid_acc = []
                avg_test_acc = []
                for experiment_number in range(num_experiments):
                    if embed == "truncated-spectral":
                        with open(f"{root_dir}"+"/Truncated_Results/"+f"highest_Validation_{dataset}_truncated_spectrumMLP_truncated_model_4.txt") as jsonFile:
                            embed_validation_acc = []
                            embed_test_acc = []
                            data = json.load(jsonFile)
                            for e in epochs:
                                embed_validation_acc.append(data[coeff][embed][str(experiment_number+1)][e][0])
                                embed_test_acc.append(data[coeff][embed][str(experiment_number+1)][e][1])
                            avg_valid_acc.append(embed_validation_acc)
                            avg_test_acc.append(embed_test_acc)
                    else:
                        with open(f"{root_dir}"+"/Results/"+f"highest_Validation_{dataset}_{model}.txt") as jsonFile:
                            embed_validation_acc = []
                            embed_test_acc = []
                            data = json.load(jsonFile)
                            for e in epochs:
                                embed_validation_acc.append(data[coeff][embed][str(experiment_number+1)][e][0])
                                embed_test_acc.append(data[coeff][embed][str(experiment_number+1)][e][1])
                            avg_valid_acc.append(embed_validation_acc)
                            avg_test_acc.append(embed_test_acc)
                avg_valid_acc = [(sum(i)/num_experiments) for i in zip(*avg_valid_acc)]
                avg_test_acc = [(sum(i)/num_experiments) for i in zip(*avg_test_acc)]
                tracker = 5
                for i in range(len(avg_valid_acc)):
                    if tracker % 10 == 0:
                        average_performance_dic[embed]["avg_valid"][tracker] = avg_valid_acc[i]
                    tracker+=5
                tracker = 5
                for i in range(len(avg_test_acc)):
                    if tracker % 10 == 0:
                        average_performance_dic[embed]["avg_test"][tracker] = avg_test_acc[i]
                    tracker+=5
                validation_acc_records.append(avg_valid_acc)
                test_acc_records.append(avg_test_acc)

            record_names = ["Validation","Test"]      
            all_records = [validation_acc_records,test_acc_records]
            font_size = 15
            with open(f"{root_dir}/Table_Results/average_performance_{dataset}_{different_model_name}_embed_{combined_name}.txt", "w") as fp:
                json.dump(average_performance_dic, fp)
            for record_index in range(len(all_records)):
                for index in range(len(all_records[record_index])):
                    plt.plot(epochs,all_records[record_index][index],label=different_embed_names[index],linestyle="-", linewidth=3.0)

                plt.xticks(condensed_epochs,rotation="vertical",fontsize=font_size)
                plt.xlabel("Epochs",fontsize=font_size)
                plt.ylabel(record_names[record_index]+" Accuracy",fontsize=font_size)
                plt.title(f"Dataset: {dataset}, Model: {different_model_name}",fontsize=font_size)
                plt.legend(fontsize = font_size)
                plt.subplots_adjust(bottom=0.2)
                plt.savefig(f"{root_dir}/Simple_Graphs/{dataset}_simpleGraph_{record_names[record_index]}_coeff_{coeff}_embed_{combined_name}_{different_model_name}.png")
                plt.close()
            
            