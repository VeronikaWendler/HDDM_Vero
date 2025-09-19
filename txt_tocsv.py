# Veornika Wendler
# conversion code
# converts a .txt file into .csv

import pandas as pd

input_file = "C:/Cluster_Github/HDDM_Vero/figures_dir_OV/macleod_cluster_out/OV_replication_For_paper_6/diagnostics/gelman_rubin.txt"  
output_file = "C:/Cluster_Github/HDDM_Vero/figures_dir_OV/macleod_cluster_out/OV_replication_For_paper_6/diagnostics/gelman_rubin_For_paper_m6.csv"

data = []
with open(input_file, "r") as file:
    for line in file:
        key, value = line.strip().split(": ") 
        data.append([key, float(value)])  # as a list

#Pandas DataFrame
df = pd.DataFrame(data, columns=["Model", "Gelman-Rubin"])
df.to_csv(output_file, index=False)
print(f"File saved as {output_file}")









