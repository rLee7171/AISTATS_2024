import json
root_dir = "/home/ryanlee/Repositories/AISTATS_2024"

dataset_name = ["Cora","CiteSeer","PubMed","WikiCs","Arxiv","Products"]

model_name = ["spectrumMLP"]

different_model_names = ["MLP"]

embedding_list = ["non-symmetric"]

coeff_list = ["2"]

epochs = ["10","20","40","80","160","200"]

num_experiments=20

number_epochs=200

for coeff in coeff_list:
    for embed in embedding_list:
            for model_index in range(len(model_name)):
                with open(f"{root_dir}/Latex_Tables/compute_times_{embed}_allDS_{different_model_names[model_index]}.tex", "w") as latex_table:
                    latex_table.write("\\begin{table}")
                    latex_table.write(f"\\caption{{Compute time and highest average test accuracy achieved on all datasets, Spectral embedding, {different_model_names[model_index]}}}")
                    latex_table.write("\\begin{center}")
                    latex_table.write("\\begin{tabular}{ | m{2cm} | m{2cm}| m{2cm} | m{2cm} }")
                    latex_table.write("\\hline")
                    latex_table.write("\\textsc{Dataset} & \\textsc{Compute Time} & \\textsc{Highest Test (Epoch Achieved)}\\\\")
                    for ds in dataset_name:
                        compute_time = 0
                        avg_test = 0
                        avg_epoch = 0
                        highest_test_list = []
                        epoch_list = []
                        with open(f"{root_dir}/compute_Embedding_Times/compute_times_{ds}.txt") as embed_times:
                            data = json.load(embed_times)
                            compute_time = round(data[embed],2)
                        with open(f"{root_dir}/Results/highest_Validation_{ds}_{model_name[model_index]}.txt") as valid_results:
                            epochs = [str(i) for i in range(1,number_epochs+1) if i%5==0]
                            data = json.load(valid_results)
                            for experiment_number in range(num_experiments):
                                cur_test = 0
                                cur_epoch = 0
                                for e in epochs:
                                    cur = data[coeff][embed][str(experiment_number+1)][e][1]
                                    if cur > cur_test:
                                        cur_test = cur
                                        cur_epoch = e
                                highest_test_list.append(cur_test)
                                epoch_list.append(int(cur_epoch))
                            avg_test = round(sum(highest_test_list)/num_experiments, 2)
                            avg_epoch = round(sum(epoch_list)/num_experiments, 2)

                        latex_table.write("\\hline ")
                        latex_table.write(f"{ds} & {compute_time} & {avg_test} ({avg_epoch})\\\\")
                    latex_table.write("\\hline")
                    latex_table.write("\\end{tabular}")
                    latex_table.write("\\end{center}")
                    latex_table.write("\\end{table}")
                        
                                
                                    
                                    
                        
