import os
import numpy as np
import matplotlib.pyplot as plt

def update_counts(class_predictions, class_n, tp, tn, fp, fn):
    if class_predictions == class_n:
        tp += class_n
        tn += 25 - class_predictions
    elif class_predictions > class_n:
        tp += class_n
        tn += 25 - class_predictions
        fp += class_predictions - class_n
    elif class_predictions < class_n:
        tp += class_predictions
        tn += 25 - class_n
        fn += class_n - class_predictions
    else:
        print("counting error")
    
    return tp, tn, fp, fn

def get_metrics(tp, tn, fp, fn, tot_predicts):
    acc = (tp + tn) / tot_predicts
    acc = float(f"{(100 * acc):.2f}")
    
    try: # tp and fp could be 0
        prec = tp / (tp + fp)
        prec = float(f"{(100 * prec):.2f}")
    except: # in which case precision is 100 but causes a zero-division error
        prec = 100
    
    try:
        sens = tp / (tp + fn)
        sens = float(f"{(100 * sens):.2f}")
    except:
        sens = 100
    
    try:
        spec = tn / (tn + fp)
        spec = float(f"{(100 * spec):.2f}")
    except:
        spec = 100
    
    try:
        f1 = 2 * prec * sens / (prec + sens)
        f1 = float(f"{(f1):.2f}")
    except:
        f1 = 100
    return acc, prec, sens, spec, f1

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
        responses = file.readlines()[1:6]
    
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
                confidence = 100
            if "reservoir" in cell.strip().lower():
                if confidence >= confidence_threshold:
                    res_predictions += 1
                else:
                    land_predictions += 1
            # the cell will be separated into three different 
            elif "water bod" in cell.strip().lower():
                if confidence >= confidence_threshold:
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
    m_res, m_bod, m_land = get_confusion_matrix(model_epochs=100, 
                                                confidence_threshold=50)
    
    # --- Prepare data for plotting ---
    labels = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"]
    reservoir_scores = list(m_res)
    water_body_scores = list(m_bod)
    land_scores = list(m_land)
    
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars
    
    # --- Create the Plot ---
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(5, 3))
    
    # Create the bars for each class, offsetting the x position
    rects1 = ax.bar(x - width, reservoir_scores, width, 
                    label="Reservoirs", color="royalblue")
    rects2 = ax.bar(x, water_body_scores, width, 
                    label="Water Bodies", color="mediumseagreen")
    rects3 = ax.bar(x + width, land_scores, width, 
                    label="Land", color="sandybrown")
    
    # Add some text for labels, title and axes ticks
    ax.set_ylabel("Score (%)", fontsize=8)
    ax.set_xlabel("Performance Metric", fontsize=8)
    ax.set_title("Model Performance Metrics by Class", fontsize=8)
    ax.set_xticks(x) # Position ticks in the center of the groups
    ax.set_xticklabels(labels)
    ax.tick_params(axis="both", which="major", labelsize=7)
    ax.legend(fontsize=5)
    
    # Set y-axis limit
    ax.set_ylim(0, 105) # Go slightly above 100 for visibility
    
    # Add value labels on top of bars (optional, can be cluttered)
    ax.bar_label(rects1, padding=3, fmt="%.1f", fontsize=7)
    ax.bar_label(rects2, padding=3, fmt="%.1f", fontsize=7)
    ax.bar_label(rects3, padding=3, fmt="%.1f", fontsize=7)
    
    fig.tight_layout() # Adjust layout
    plt.show()
