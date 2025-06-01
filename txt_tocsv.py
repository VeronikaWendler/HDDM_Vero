# Veornika Wendler
# conversion code
# converts a .txt file into .csv

import pandas as pd

input_file = "/home/jovyan/OfficialTutorials/THESIS_HDDM/figures_garcia/garcia_replication_EE_5/diagnostics/gelman_rubin.txt"  # Adjust the path
output_file = "/home/jovyan/OfficialTutorials/THESIS_HDDM/figures_garcia/garcia_replication_EE_5/diagnostics/gelman_rubin_EE_m5.csv"

data = []
with open(input_file, "r") as file:
    for line in file:
        key, value = line.strip().split(": ") 
        data.append([key, float(value)])  # as a list

#Pandas DataFrame
df = pd.DataFrame(data, columns=["Model", "Gelman-Rubin"])
df.to_csv(output_file, index=False)
print(f"File saved as {output_file}")
