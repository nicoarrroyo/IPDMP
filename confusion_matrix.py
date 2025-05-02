def get_metrics(class_predictions, class_n, tp, tn, fp, fn):
    if class_predictions == class_n:
        tp += class_n
        tn += 25 - class_predictions
    if class_predictions > class_n:
        tp += class_n
        tn += 25 - class_predictions
        fp += class_predictions - class_n
    if class_predictions < class_n:
        tp += class_predictions
        tn += 25 - class_n
        fn += class_n - class_predictions
    return tp, tn, fp, fn

import os

os.chdir(
    os.path.join(
        "C:\\", "Users", "nicol", "Documents", "UoM", "YEAR 3", 
        "Individual Project", "Downloads", "Sentinel 2", 
        "S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU_20250301T152054.SAFE"
        )
    )

with open("responses_5000_chunks.csv", mode="r") as file:# responses file
    responses = file.readlines()[1:6]

with open ("P_5000_50_T31UCU.csv", mode="r") as file: # predictions file
    predictions = file.readlines()[2:len(responses)+2]

tp_res = 0 # true positive
tn_res = 0 # true negative
fp_res = 0 # false positive
fn_res = 0 # false negative

tp_bod = 0 # true positive
tn_bod = 0 # true negative
fp_bod = 0 # false positive
fn_bod = 0 # false negative

tp_land = 0 # true positive
tn_land = 0 # true negative
fp_land = 0 # false positive
fn_land = 0 # false negative

res_rows = []
bod_rows = []
land_rows = []

for i, response in enumerate(responses):
    split_response = response.split(",")
    chunk_n = int(split_response[0])
    
    res_n = int(split_response[1])
    bod_n = int(split_response[2])
    # each of the 25 minichunks that isn't res or bod must be land
    land_n = 25 - res_n - bod_n
    
    prediction = predictions[i].split(",")
    res_predictions = 0
    bod_predictions = 0
    land_predictions = 0
    for j, cell in enumerate(prediction):
        if "reservoir" in cell.strip().lower():
            res_predictions += 1
        elif "water bod" in cell.strip().lower():
            bod_predictions += 1
        elif "land" in cell.strip().lower():
            land_predictions += 1
    
    if res_predictions == res_n:
        tp_res += res_n
        tn_res += 25 - res_predictions
    if res_predictions > res_n:
        tp_res += res_n
        tn_res += 25 - res_predictions
        fp_res += res_predictions - res_n
    if res_predictions < res_n:
        tp_res += res_predictions
        tn_res += 25 - res_n
        fn_res += res_n - res_predictions
    
    if bod_predictions == bod_n:
        tp_bod += bod_n
        tn_bod += 25 - bod_predictions
    if bod_predictions > bod_n:
        tp_bod += bod_n
        tn_bod += 25 - bod_predictions
        fp_bod += bod_predictions - bod_n
    if bod_predictions < bod_n:
        tp_bod += bod_predictions
        tn_bod += 25 - bod_n
        fn_bod += bod_n - bod_predictions
    
    if land_predictions == land_n and land_n > 0:
        tp_land += land_n
        tn_land += 25 - land_predictions
    if land_predictions > land_n:
        tp_land += land_n
        tn_land += 25 - land_predictions
        fp_land += land_predictions - land_n
    if land_predictions < land_n:
        tp_land += land_predictions
        tn_land += 25 - land_n
        fn_land += land_n - land_predictions
    
    
    

