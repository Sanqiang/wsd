"""
Merge sense invventory of 3 TestSets.
"""
from preprocess.file_helper import json_reader
import csv


if __name__ == '__main__':
    data_path = "/home/luoz3/wsd_data"
    msh_processed_path = data_path + "/msh/msh_processed"
    share_processed_path = data_path + "/share/processed"
    # umn_processed_path = data_path + "/umn/umn_processed"

    sense_inventory_msh = json_reader(msh_processed_path + "/MSH_sense_inventory_one_word.json")
    sense_inventory_share = json_reader(share_processed_path + "/all_sense_inventory.json")
    # sense_inventory_umn = json_reader(umn_processed_path+"/UMN_sense_cui_inventory.json")

    sense_inventory_merge = {}
    for abbr, cui_list in sense_inventory_share.items():
        sense_inventory_merge[abbr] = set(cui_list)
    for abbr, cui_list in sense_inventory_msh.items():
        if abbr in sense_inventory_merge:
            sense_inventory_merge[abbr].update(cui_list)
        else:
            sense_inventory_merge[abbr] = set(cui_list)
    # remove abbr with only 1 sense
    sense_inventory_merge_ambiguous = {}
    for abbr, cui_set in sense_inventory_merge.items():
        if len(cui_set)>1:
            sense_inventory_merge_ambiguous[abbr] = cui_set

    # write to csv file
    with open(data_path+"/sense_inventory_share_and_msh.csv","w") as file:
        fieldnames = ['abbr', 'cui']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for abbr, cui_set in sense_inventory_merge_ambiguous.items():
            for cui in cui_set:
                writer.writerow({
                    "abbr": abbr,
                    "cui": cui
                })

    print()
