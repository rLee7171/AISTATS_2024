import json
import statistics
import matplotlib.pyplot as plt
import numpy as np
root_dir = "/home/ryanlee/Repositories/AISTATS_2024"

dataset_name = ["Cora","CiteSeer","PubMed","WikiCs","Arxiv","Products"]

model_name = ["conicMLP_model_1","conicMLP_model_4"]

different_model_names = ["Radial_model_1","conicMLP_model_4"]

abbr_names = ["RA_1", "RA_4"]

embedding_list = ["non-symmetric", "symmetric","deepwalk"]

coeff_list = ["2"]
number_epochs=200
epochs = [str(i) for i in range(1,number_epochs+1) if i%5==0]
condensed_epochs = [str(i) for i in range(1,number_epochs+1) if i%20==0]
num_experiments=20

for dataset in dataset_name:
    average_performance_dic = {}
    for coeff in coeff_list:
        for embed in embedding_list:
            validation_acc_records = []
            test_acc_records = []
            for model in model_name:
                average_performance_dic[model] = {}
                average_performance_dic[model]["avg_valid"] = {}
                average_performance_dic[model]["avg_test"] = {}
                avg_valid_acc = []
                avg_test_acc = []
                for experiment_number in range(num_experiments):
                    with open(f"{root_dir}"+"/Results/"+f"highest_Validation_{dataset}_{model}.txt") as jsonFile:
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
                tracker = 5
                for i in range(len(avg_valid_acc)):
                    if tracker % 10 == 0:
                        average_performance_dic[model]["avg_valid"][tracker] = avg_valid_acc[i]
                    tracker+=5
                tracker = 5
                for i in range(len(avg_test_acc)):
                    if tracker % 10 == 0:
                        average_performance_dic[model]["avg_test"][tracker] = avg_test_acc[i]
                    tracker+=5
                validation_acc_records.append(avg_valid_acc)
                test_acc_records.append(avg_test_acc)

            record_names = ["Validation","Test"]      
            all_records = [validation_acc_records,test_acc_records]
            font_size = 15
            with open(f"{root_dir}/Table_Results/average_performance_{dataset}_{embed}_{abbr_names[0]}_{abbr_names[1]}.txt", "w") as fp:
                json.dump(average_performance_dic, fp)
            for record_index in range(len(all_records)):
                for index in range(len(all_records[record_index])):
                    plt.plot(epochs,all_records[record_index][index],label=different_model_names[index],linestyle="-", linewidth=3.0)

                plt.xticks(condensed_epochs,rotation="vertical",fontsize=font_size)
                plt.xlabel("Epochs",fontsize=font_size)
                plt.ylabel(record_names[record_index]+" Accuracy",fontsize=font_size)
                plt.title(f"Dataset: {dataset}, Embed: {embed}",fontsize=font_size)
                plt.legend(fontsize = font_size)
                plt.subplots_adjust(bottom=0.2)
                plt.savefig(f"{root_dir}/Simple_Graphs/{dataset}_simpleGraph_{record_names[record_index]}_coeff_{coeff}_embed_{embed}.png")
                plt.close()
            
            