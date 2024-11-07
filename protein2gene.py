import openpyxl
import pickle

workbook = openpyxl.load_workbook('/home/yuanrz/pdb_path/process_data_liyaxuan/protein2gene.xlsx')

excel_dict = {}

sheet = workbook.active
row_index = 0
for row in sheet.rows:
    if row_index > 0:  
        key = row[1].value  
        value = row[0].value  
        excel_dict[key] = value  
    row_index += 1


import os
import shutil

source_folder = "./struct/proteins_edgs"
target_folder = "../final_data/struct_features"

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

unmatched_files = []

for filename in os.listdir(source_folder):
    if filename.endswith(".txt"):
        entry = filename.split(".")[0]
        
        if entry in excel_dict.values():
            for gene, entry_value in excel_dict.items():
                if entry_value == entry:
                    new_filename = gene + ".txt"
                    source_file_path = os.path.join(source_folder, filename)
                    target_file_path = os.path.join(target_folder, new_filename)
                    shutil.copyfile(source_file_path, target_file_path)
                    print(f"Copied {filename} to {new_filename}")
                    break
        else:
            unmatched_files.append(filename)
            print(f"No corresponding gene found for {filename}")

print("Unmatched files:", unmatched_files)