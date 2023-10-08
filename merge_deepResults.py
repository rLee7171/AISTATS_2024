import json

root_dir = "/home/ryanlee/Repositories/AISTATS_2024"

dataset_name = ["Cora","CiteSeer","PubMed","WikiCs","Arxiv"]

model_name = ["conicMLP_model_0","conicMLP_model_1","conicMLP_model_2","conicMLP_model_3","conicMLP_model_4","spectrumMLP"]

embedding_list = ["non-symmetric", "symmetric","deepwalk"]

coeff_list = ["2"]
number_epochs=200
epochs = [str(i) for i in range(1,number_epochs+1) if i%5==0]
condensed_epochs = [str(i) for i in range(1,number_epochs+1) if i%20==0]
num_experiments=20

for dataset in dataset_name:
    for model in model_name:
        with open(f"{root_dir}"+"/Results/"+f"highest_Validation_{dataset}_{model}.txt") as jsonFile1:
                data1 = json.load(jsonFile1)
        with open(f"{root_dir}"+"/Deepwalk_Results/"+f"highest_Validation_{dataset}_{model}_Deepwalk.txt") as jsonFile2:
                data2 = json.load(jsonFile2)
        for key1 in data1.keys():
            for key2 in data1[key1].keys():
                 for key3 in data1[key1][key2].keys():
                      for key4 in data1[key1][key2][key3].keys():
                           if key2 == "deepwalk":
                                data1[key1][key2][key3][key4] = data2[key1][key2][key3][key4]


        with open(f"{root_dir}/Merged_Results/highest_Validation_{dataset}_{model}.txt", "w") as fp:
            json.dump(data1, fp)