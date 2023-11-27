import json
root_dir = "/home/ryanlee/Repositories/AISTATS_2024"

dataset_name = ["Cora","CiteSeer","PubMed","WikiCs","Arxiv","Products"]

model_name = ["conicMLP_model_1"]

different_model_name = "Radial"

embedding_list = ["non-symmetric","truncated-spectral"]

different_embed_names = ["Spectral", "Truncated"]

combined_name = "truncated_spec_radial"

coeff_list = ["2"]

epochs = ["10","20","40","80","160","200"]

truncated_iterations = "32"

num_experiments=20

number_epochs=200

for coeff in coeff_list:
    for model_index in range(len(model_name)):
            with open(f"{root_dir}/Latex_Tables/compute_times_{combined_name}_allDS.tex", "w") as latex_table:
                latex_table.write("\\begin{table}")
                latex_table.write(f"\\caption{{Compute time and highest average test accuracy achieved on all datasets, {embedding_list[0]} and {embedding_list[1]} embeddings, {different_model_name}}}")
                latex_table.write("\\begin{center}")
                latex_table.write("\\begin{tabular}{ | m{2cm} | m{2cm}| m{2cm} | m{2cm} }")
                latex_table.write("\\hline")
                latex_table.write("\\textsc{Dataset} & \\textsc{Embedding} & \\textsc{Compute Time} & \\textsc{Highest Test (Epoch Achieved)}\\\\")
                for ds in dataset_name:
                    for embed in embedding_list:
                        compute_time = 0
                        avg_test = 0
                        avg_epoch = 0
                        highest_test_list = []
                        epoch_list = []
                        with open(f"{root_dir}/compute_Embedding_Times/compute_times_{ds}.txt") as embed_times:
                            data = json.load(embed_times)
                            if embed == "truncated-spectral":
                                compute_time = round(data[embed][truncated_iterations],2)
                            else:
                                compute_time = round(data[embed],2)

                        if embed == "truncated-spectral":
                            with open(f"{root_dir}/Truncated_Results/highest_Validation_{ds}_{model_name[model_index]}_truncated_model_4.txt") as valid_results:
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

                        else:
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
                        latex_table.write(f"{ds} & {embed} & {compute_time} & {avg_test} ({avg_epoch})\\\\")
                latex_table.write("\\hline")
                latex_table.write("\\end{tabular}")
                latex_table.write("\\end{center}")
                latex_table.write("\\end{table}")