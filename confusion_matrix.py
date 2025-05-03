import os
import matplotlib.pyplot as plt
import numpy as np

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
    
    # to prevent "division by zero" errors
    if tp == 0:
        tp += 1
    if tn == 0:
        tn += 1
    if fp == 0:
        fp += 1
    if fn == 0:
        fn += 1
    
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

def get_confusion_matrix(model_epochs, confidence_threshold):
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
    
    for i, response in enumerate(responses):
        split_response = response.split(",")
        
        res_n = int(split_response[1])
        bod_n = int(split_response[2])
        # each of the 25 minichunks that isn't res or bod must be land
        land_n = 25 - res_n - bod_n
        
        prediction = predictions[i].split(",")
        res_predictions = 0
        bod_predictions = 0
        land_predictions = 0
        for j, cell in enumerate(prediction):
            try:
                confidence = float(cell.split(" ")[1])
            except:
                continue
            if "reservoir" in cell.strip().lower():
                if confidence > confidence_threshold:
                    res_predictions += 1
                else:
                    land_predictions += 1
            elif "water bod" in cell.strip().lower():
                if confidence > confidence_threshold:
                    bod_predictions += 1
                else:
                    land_predictions += 1
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
    #for C in range(30, 60, 10): # anything below C = 30 makes no difference
    metrics_res = []
    metrics_bod = []
    metrics_land = []
    
    #epoch_options = [50, 55, 60, 65, 70]
    epoch_options = [50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70]
    for epoch_option in epoch_options:
        m_res, m_bod, m_land = get_confusion_matrix(epoch_option, 
                                                    confidence_threshold=100)
        metrics_res.append(m_res)
        metrics_bod.append(m_bod)
        metrics_land.append(m_land)
    
    accuracies = [metrics_bod[i][0] for i in range(len(epoch_options))]
    plt.plot(epoch_options, accuracies, label="Water Bodies", linewidth=1)
    accuracies = [metrics_res[i][0] for i in range(len(epoch_options))]
    plt.plot(epoch_options, accuracies, label="Reservoirs", linewidth=1)
    plt.title("Epoch vs Accuracies")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=5)
    plt.grid(True)
    plt.show()
    
    precisions_bod = [metrics_bod[i][1] for i in range(len(epoch_options))]
    precisions_res = [metrics_res[i][1] for i in range(len(epoch_options))]
    plt.plot(epoch_options, precisions_bod, label="Water Bodies", linewidth=1)
    plt.plot(epoch_options, precisions_res, label="Reservoirs", linewidth=1)
    plt.title("Epoch vs Precisions")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend(fontsize=5)
    plt.grid(True)
    #plt.show()
    
    sensitivities_bod = [metrics_bod[i][2] for i in range(len(epoch_options))]
    sensitivities_res = [metrics_res[i][2] for i in range(len(epoch_options))]
    #plt.plot(epoch_options, sensitivities_bod, label="Water Bodies", linewidth=1)
    plt.plot(epoch_options, sensitivities_res, label="Reservoirs", linewidth=1)
    plt.title("Epoch vs Sensitivities")
    plt.xlabel("Epoch")
    plt.ylabel("Sensitivity")
    plt.legend(fontsize=5)
    plt.grid(True)
    #plt.show()
    
    specificities = [metrics_bod[i][3] for i in range(len(epoch_options))]
    plt.plot(epoch_options, specificities, label="Water Bodies", linewidth=1)
    specificities = [metrics_res[i][3] for i in range(len(epoch_options))]
    plt.plot(epoch_options, specificities, label="Reservoirs", linewidth=1)
    plt.title("Epoch vs Specificities")
    plt.xlabel("Epoch")
    plt.ylabel("Specificity")
    plt.legend(fontsize=5)
    plt.grid(True)
    #plt.show()
    
    # the threshold that maximises this plot is ideal
    precisions_bod = np.array(precisions_bod)
    sensitivities_bod = np.array(sensitivities_bod)
    f1_scores = 2 * precisions_bod * sensitivities_bod / (precisions_bod + 
                                                          sensitivities_bod)
    #plt.plot(epoch_options, f1_scores, label="Water Bodies")
    precisions_res = np.array(precisions_res)
    sensitivities_res = np.array(sensitivities_res)
    f1_scores = 2 * precisions_res * sensitivities_res / (precisions_res + 
                                                          sensitivities_res)
    plt.plot(epoch_options, f1_scores, label="Reservoirs", linewidth=1)
    plt.title("Epoch vs F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend(fontsize=5)
    plt.grid(True)
    plt.show()

