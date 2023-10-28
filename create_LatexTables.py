import json
root_dir = "/home/ryanlee/Repositories/AISTATS_2024"

dataset_name = ["Cora","CiteSeer","PubMed","WikiCs","Arxiv","Products"]

model_name = ["conicMLP_model_0","spectrumMLP"]

different_model_names = ["Radial_model_0", "MLP"]

performance_names = ["avg_valid", "avg_test"]

embedding_list = ["non-symmetric"]

coeff_list = ["2"]

epochs = ["10","20","40","80","160","200"]

for per_name in performance_names:
    for embed in embedding_list:
        with open(f"{root_dir}/Latex_Tables/{per_name}_allDS_{embed}.tex", "w") as latex_table:
            latex_table.write("\begin{table}[ht]")
            latex_table.write("\caption{Average validation accuracy overtime for \textsc{Radial\_model\_0} on all datasets, non-symmetric embedding")
            latex_table.write("\begin{center}")
            latex_table.write("\scalebox{0.92}{")
            latex_table.write("\begin{tabular}{lllllllllll}")
            latex_table.write("\textsc{Epoch}&10&20&40&80&160&200\\")
            latex_table.write("\hline\\")
            for dataset in dataset_name:
                with open(f"{root_dir}"+"/Table_Results/"+f"average_performance_{dataset}_{embed}.txt") as jsonFile:
                    data = json.load(jsonFile)
                    for model_index in range(len(model_name)):
                        latex_table.write("\textsc{Dataset}&"+ dataset +"&&&&&\\")
                        results = []
                        for e in epochs:
                            results.append(round(data[model_name[model_index]][per_name][e], 2))
                        latex_table.write("\textsc{"+different_model_names[model_index]+
                                          "} & " + f"{results[0]}" +
                                          " & " + f"{results[1]}" +
                                          " & " + f"{results[2]}" +
                                          " & " + f"{results[3]}" +
                                          " & " + f"{results[4]}" +
                                          " & " + f"{results[5]}")
                        latex_table.write("\\")
                    latex_table.write("\\")
                    latex_table.write("\hline")
                    latex_table.write("\\")
            latex_table.write("\end{tabular}}")
            latex_table.write("\end{center}")
            latex_table.write("\end{table}")



                

    