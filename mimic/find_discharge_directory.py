import os
import os.path as path

root_dir = "/data"

candidate_dir_names = set(["7", "711", "712", "713", "714", "715", "716", "717", "719", "721", "722", "723", "724", "725", "726", "729", "731", "732", "733", "734", "735", "740", "741", "745", "746", "747", "750", "751", "752", "759", "764", "771", "783", "786", "787", "795", "797", "798", "799", "7911", "188425", "CHRUDischargeSummaryAttending", "DischargeSummary", "DischargeSummaryAttending", "HospDC", "HospDC", "HospitalDischargeSummary", "HospitalDischargeSummary", "NeonatologyDischargeSummary", "SelectDischargeSummary", "21", "XRSPCM4VFLEXEXT", "XRSPECIMEN"])

def recursive_find_directories(parent_dir_path,result):
    for dir_name in os.listdir(parent_dir_path):
        full_dir_path = path.join(parent_dir_path, dir_name)
        if not path.isdir(full_dir_path):
            continue
        if dir_name in candidate_dir_names:
            print(full_dir_path)
            recursive_find_discharge_file(full_dir_path, result)
        else:
            recursive_find_directories(full_dir_path,result)

def recursive_find_discharge_file(parent_dir_path, result):
    global counter
    if path.isdir(parent_dir_path):
        for dir_name in os.listdir(parent_dir_path):
            full_dir_path = path.join(parent_dir_path, dir_name)
            recursive_find_discharge_file(full_dir_path,result)
    else:
        result.append(parent_dir_path)



all_discharge_file_paths = []
recursive_find_directories(root_dir, all_discharge_file_paths)

## this is all discharge files
print(all_discharge_file_paths)


# print(result)