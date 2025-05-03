import os
import matplotlib.pyplot as plt

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
    acc = float(f"{(100*acc):.2f}")
    
    prec = tp / (tp + fp)
    prec = float(f"{(100*prec):.2f}")
    
    sensitivity = tp / (tp + fn)
    sensitivity = float(f"{(100*sensitivity):.2f}")
    
    specificity = tn / (tn + fp)
    specificity = float(f"{(100*specificity):.2f}")
    return acc, prec, sensitivity, specificity

def get_confusion_matrix(model_epochs):
    os.chdir(
        os.path.join(
            "C:\\", "Users", "nicol", "Documents", "UoM", "YEAR 3", 
            "Individual Project", "Downloads", "Sentinel 2", 
            "S2C_MSIL2A_20250331T110651_N0511_R137_T31UCU_20250331T143812.SAFE"
            )
        )
    
    # responses file
    with open("responses_5000_chunks.csv", mode="r") as file:
        responses = file.readlines()[1:1650]
    
    # predictions file
    with open (f"P_5000_{model_epochs}_T31UCU.csv", mode="r") as file:
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
    
    metrics_res = get_metrics(tp_res, tn_res, 
                              fp_res, fn_res, 
                              (25*len(predictions)))
    
    metrics_bod = get_metrics(tp_bod, tn_bod, 
                              fp_bod, fn_bod, 
                              (25*len(predictions)))
    
    metrics_land = get_metrics(tp_land, tn_land, 
                              fp_land, fn_land, 
                              (25*len(predictions)))
    
    return metrics_res, metrics_bod, metrics_land
    

if __name__ == "__main__":
    metrics_res = []
    metrics_bod = []
    metrics_land = []
    
    epoch_options = [50, 55, 60, 65, 70]
    for epoch_option in epoch_options:
        m_res, m_bod, m_land = get_confusion_matrix(epoch_option)
        metrics_res.append(m_res)
        metrics_bod.append(m_bod)
        metrics_land.append(m_land)
    
    accuracies = [metrics_bod[i][0] for i in range(5)]
    plt.plot(epoch_options, accuracies)
    accuracies = [metrics_res[i][0] for i in range(5)]
    plt.plot(epoch_options, accuracies)
    plt.title("epochs vs accuracies")
    plt.show()
    
    precisions = [metrics_bod[i][1] for i in range(5)]
    plt.plot(epoch_options, precisions)
    precisions = [metrics_res[i][1] for i in range(5)]
    plt.plot(epoch_options, precisions)
    plt.title("epochs vs precisions")
    plt.show()
    
    sensitivities = [metrics_bod[i][2] for i in range(5)]
    plt.plot(epoch_options, sensitivities)
    sensitivities = [metrics_res[i][2] for i in range(5)]
    plt.plot(epoch_options, sensitivities)
    plt.title("epochs vs sensitivities")
    plt.show()
    
    specificities = [metrics_bod[i][3] for i in range(5)]
    plt.plot(epoch_options, specificities)
    specificities = [metrics_res[i][3] for i in range(5)]
    plt.plot(epoch_options, specificities)
    plt.title("epochs vs specificities")
    plt.show()
    
# =============================================================================
#     for i in range(5):
#         plt.plot(metrics_bod[i], epoch_options[i])
#     plt.show()
#     for i in range(5):
#         plt.plot(metrics_res[i], epoch_options[i])
#     plt.show()
#     for i in range(5):
#         plt.plot(metrics_land[i], epoch_options[i])
#     plt.show()
# =============================================================================

