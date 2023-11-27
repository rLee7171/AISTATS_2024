import json
root_dir = "/home/ryanlee/Repositories/AISTATS_2024"

dataset_name = ["Cora","CiteSeer","PubMed","WikiCs","Arxiv","Products"]

model_name = ["conicMLP_model_1"]

different_model_names = ["Radial$_{2,2}$"]

performance_names = ["avg_valid", "avg_test"]

embedding_list = ["non-symmetric", "deepwalk"]

different_embedding_names = ["Spectral", "Deepwalk"]

coeff_list = ["2"]

combined_embedding_name = "deepwalk_spec"

epochs = ["10","20","40","80","160","200"]

for per_name in performance_names:
    for model_index in range(len(model_name)):
        with open(f"{root_dir}/Latex_Tables/{per_name}_allDS_{combined_embedding_name}_RA_1.tex", "w") as latex_table:
            latex_table.write("\\begin{table}[ht]")
            if per_name == "avg_valid":
                latex_table.write(f"\\caption{{Average validation accuracy overtime on all datasets, Radial$_{2,2}$ model}}")
            else:
                latex_table.write(f"\\caption{{Average test accuracy overtime on all datasets, Radial$_{2,2}$ model}}")

            latex_table.write("\\begin{center}")
            latex_table.write("\\scalebox{0.82}{")
            latex_table.write("\\begin{tabular}{lllllllllll}")
            latex_table.write("\\textsc{Epoch}&10&20&40&80&160&200\\\\")
            latex_table.write("\\hline\\\\")
            for dataset in dataset_name:
                for embed_index in range(len(embedding_list)):
                    with open(f"{root_dir}"+"/Table_Results/"+f"average_performance_{dataset}_{embedding_list[embed_index]}.txt") as jsonFile:
                        data = json.load(jsonFile)
                        results = []
                        for e in epochs:
                            results.append(int(round(data[model_name[model_index]][per_name][e], 2)*100))
                        latex_table.write("\\textsc{"+dataset+"-"+different_embedding_names[embed_index]+
                                    "} & " + f"{results[0]}\\%" +
                                    " & " + f"{results[1]}\\%" +
                                    " & " + f"{results[2]}\\%" +
                                    " & " + f"{results[3]}\\%" +
                                    " & " + f"{results[4]}\\%" +
                                    " & " + f"{results[5]}\\%")
                        latex_table.write("\\\\")
                latex_table.write("\\\\")
                latex_table.write("\\hline")
                latex_table.write("\\\\")
            latex_table.write("\\end{tabular}}")
            latex_table.write("\\end{center}")
            latex_table.write("\\end{table}")



                

    