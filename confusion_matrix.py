def update_counts(class_predictions, class_n, tp, tn, fp, fn):
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

def get_metrics(tp, tn, fp, fn, tot_predicts):
    acc = (tp + tn) / tot_predicts
    prec = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return acc, prec, sensitivity, specificity

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
    
    tp_res, tn_res, fp_res, fn_res = update_counts(res_predictions, 
                                                   res_n, 
                                                   tp_res, tn_res, 
                                                   fp_res, fn_res)
    
    tp_bod, tn_bod, fp_bod, fn_bod = update_counts(bod_predictions, 
                                                   bod_n, 
                                                   tp_bod, tn_bod, 
                                                   fp_bod, fn_bod)
    
    tp_land, tn_land, fp_land, fn_land = update_counts(land_predictions, 
                                                       land_n, 
                                                       tp_land, tn_land, 
                                                       fp_land, fn_land)

metrics = get_metrics(tp_res, tn_res, 
                      fp_res, fn_res, 
                      (25*len(predictions)))
acc_res, prec_res, sensitivity_res, specificity_res = metrics

metrics = get_metrics(tp_bod, tn_bod, 
                      fp_bod, fn_bod, 
                      (25*len(predictions)))
acc_bod, prec_bod, sensitivity_bod, specificity_bod = metrics

metrics = get_metrics(tp_land, tn_land, 
                      fp_land, fn_land, 
                      (25*len(predictions)))
acc_land, prec_land, sensitivity_land, specificity_land = metrics

