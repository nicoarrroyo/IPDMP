# %% 0. Start
# %%% i. Import External Libraries
import time
MAIN_START_TIME = time.monotonic()
import os
import numpy as np
import re # for parsing filenames
import sys
import math
import matplotlib.pyplot as plt

# %%% ii. Import Internal Functions
from KRISP import run_model
from data_handling import check_file_permission, blank_entry_check

# %%% iii. Directory Management
try: # personal pc mode
    HOME = os.path.join("C:\\", "Users", "nicol", "Documents", "UoM", "YEAR 3", 
                        "Individual Project", "Downloads")
    os.chdir(HOME)
except: # uni mode
    HOME = os.path.join("C:\\", "Users", "c55626na", "OneDrive - "
                        "The University of Manchester", "Individual Project")
    os.chdir(HOME)

class_names = ["land", "reservoirs", "water bodies"]

# %% prelim
# do it row by row
folder = ("S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU_20250301T152054.SAFE")
n_chunks = 5000
model_epochs = 1000
n_chunk_preds = 200

predictions_file = f"preds_{str(n_chunks)}chunks_{model_epochs}epochs.csv"

minichunk_header = ",minichunks,"
chunk_header = "chunk," + ",".join(map(str, range(25)))

os.chdir(os.path.join(HOME, "Sentinel 2", folder))

# %% find biggest chunk
check_file_permission(predictions_file)
blank_entry_check(predictions_file)

with open(predictions_file, mode="r") as file:
    lines = file.readlines()

biggest_chunk = -1
for i, line in enumerate(lines):
    if i < 2:
        continue # skip first couple rows, otherwise it's not worth saving
    try:
        biggest_chunk = max(biggest_chunk, int(line.split(",")[0])) + 1
    except:
        continue

print("CONTINUING FROM:")
print(f"chunk {biggest_chunk} out of {n_chunks}")
print(f"file {biggest_chunk * 25} out of {n_chunks * 25}")
print(f"\nCOMPLETION: {100 * biggest_chunk / n_chunks}%")
print("\nTHIS RUN WILL MAKE PREDICTIONS ON:")
print(f"{n_chunk_preds} chunks")
print(f"{n_chunk_preds * 25} files")

# %% yield predictions
the_results = run_model(
    folder=folder, 
    n_chunks=n_chunks, 
    model_name=f"ndwi model epochs-{model_epochs}.keras", 
    max_multiplier=0.41, 
    plot_examples=False, 
    start_chunk=biggest_chunk, 
    n_chunk_preds=int(n_chunk_preds)
    )

# %% find biggest chunk
check_file_permission(predictions_file)
blank_entry_check(predictions_file)

if biggest_chunk < 1:
    with open(predictions_file, mode="a") as ap:
        ap.write(minichunk_header)
        ap.write(f"\n{chunk_header}")

# %% write the results
with open(predictions_file, mode="a") as ap:
    for result in the_results:
        chunk_num, minichunk_num, label, confidence = result
        base_entry = f"{label} {str(confidence)},"
        if minichunk_num == 0:
            ap.write(f"\n{str(chunk_num)},{base_entry}")
        else:
            ap.write(f"{base_entry}")

check_file_permission(predictions_file)
blank_entry_check(predictions_file)

print("\nPREDICTIONS COMPLETE UP TO: ")
print(f"chunk {(biggest_chunk + n_chunk_preds)} out of {n_chunks}")
print(f"file {int(biggest_chunk + n_chunk_preds) * 25} out of {n_chunks * 25}")
print(f"\nCOMPLETION: {100 * (biggest_chunk + n_chunk_preds) / n_chunks}%")
