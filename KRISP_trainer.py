# %% 0.Start
""" KRISP Trainer
Keras Reservoir Identification Sequential Platform Trainer
"""
print("=== Script Start ===")
# %%% i. Import External Libraries
import time
MAIN_START_TIME = time.monotonic()
import pathlib
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# %%% ii. Import Internal Functions
from image_handling import image_to_array
from user_interfacing import start_spinner, end_spinner
print("Imports complete.")

# %% 1. Configuration
print("=== 1. Configuring Parameters ===")
# --- Core Settings ---
MODEL_TYPE = "ndwi" # Options: "ndwi", "tci"

HOME = os.path.dirname(os.getcwd()) # HOME path is one level up from the cwd
BASE_PROJECT_DIR = os.path.join(HOME, "Downloads")

SENTINEL_FOLDER = ("S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU"
                   "_20250301T152054.SAFE")
DATA_BASE_PATH = os.path.join(BASE_PROJECT_DIR, "Sentinel 2", 
                              SENTINEL_FOLDER, "training data")
TRAINING_DATA_PATH = os.path.join(DATA_BASE_PATH, "training_data.tfrecord")
# contains "reservoirs", "water bodies", "land", and "sea"
img_features = { # ensure this matches create_tf_example from data handling
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'label_text': tf.io.FixedLenFeature([], tf.string),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

# --- Training Parameters ---
EPOCHS = 40
LEARNING_RATE = 0.001 # Adam optimizer default, but can be specified

# --- Output Settings ---
SAVE_MODEL = False # Set to True to save the trained model
MODEL_SAVE_DIR = os.path.join(BASE_PROJECT_DIR, "saved_models")
MODEL_FILENAME = f"{MODEL_TYPE} model epochs-{EPOCHS}.keras"

# --- Model Parameters ---
DROPOUT_RATE = 0.2

# --- Test Image ---
# Ensure this path is relative to DATA_BASE_PATH or provide a full path
TEST_IMAGE_NAME = f"{MODEL_TYPE} chunk 1 reservoir 1.png"

# --- Dataset Parameters ---
IMG_HEIGHT = int(157/5) # must adjust this for the actual image size!!!!
IMG_WIDTH = int(157/5)
BATCH_SIZE = 1024
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 123 # For reproducibility of splits
CLASS_NAMES = ["land", "reservoirs", "sea", "water bodies"]
num_classes = len(CLASS_NAMES)
print(f"Image size: ({IMG_HEIGHT}, {IMG_WIDTH})")
print(f"Batch size: {BATCH_SIZE}")
print(f"Validation split: {VALIDATION_SPLIT}")
print(f"{len(CLASS_NAMES)} classes: {CLASS_NAMES}")
def parse_img(example_proto):
    features = tf.io.parse_single_example(
        example_proto, 
        img_features)
    label = features["label"]
    img = tf.io.decode_png(features["image_raw"], channels=3)
    
    height = tf.cast(features["height"], tf.int32)
    width = tf.cast(features["width"], tf.int32)
    depth = tf.cast(features["depth"], tf.int32)
    
    img = tf.reshape(img, [height, width, depth])
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return img, label

# %% 2. Prepare Paths and Directories
print("=== 2. Preparing Paths ===")
data_dir = os.path.join(DATA_BASE_PATH, MODEL_TYPE)
model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
test_image_path = os.path.join(DATA_BASE_PATH, TEST_IMAGE_NAME)

# Create output directories if they don't exist
if SAVE_MODEL:
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print("Model will be saved to nominal directory")

# --- Path Validation ---
if not os.path.exists(TRAINING_DATA_PATH):
    print(f"Error: Data directory not found at {TRAINING_DATA_PATH}")
    print("Please check the configuration for BASE_PROJECT_DIR, "
          "SENTINEL_FOLDER, and MODEL_TYPE.")
    sys.exit(1)
else:
    print("Using nominal data directory.")

if not os.path.exists(test_image_path):
    print(f"WARNING: Test image not found at {test_image_path}. Prediction "
           "step will fail or use incorrect data.")
else:
    print(f"Using nominal test image: {TEST_IMAGE_NAME}")

data_dir_pathlib = pathlib.Path(TRAINING_DATA_PATH)
try:
    image_count = len(list(data_dir_pathlib.glob('*/*.png')))
    print(f"Found {image_count} images in nominal data directory.")
    if image_count == 0:
        print("WARNING: No images found. Check the data directory structure "
              "and image format.")
except Exception as e:
    print(f"Error counting images: {e}")
    image_count = 0 # Assume zero if listing fails

if image_count < BATCH_SIZE:
    print(f"WARNING: Total image count ({image_count}) is less than the batch "
           f"size ({BATCH_SIZE}). This might cause issues during training.")

# %% 3. Prepare the Dataset
print("=== 3. Loading and Preparing Dataset ===")

raw_img_dataset = tf.data.TFRecordDataset(TRAINING_DATA_PATH)

dataset_size = sum(1 for _ in raw_img_dataset)
print(f"found {dataset_size} records in TFRecord file")

shuffled_dataset = raw_img_dataset.shuffle(
    buffer_size=dataset_size, 
    seed=RANDOM_SEED, 
    reshuffle_each_iteration=True)

val_size = int(dataset_size * VALIDATION_SPLIT)
val_ds = shuffled_dataset.take(val_size)
train_ds = shuffled_dataset.skip(val_size)

val_ds = val_ds.map(parse_img, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(parse_img, num_parallel_calls=tf.data.AUTOTUNE)

val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

if num_classes <= 1:
    print("Error: Need at least two classes for classification.")
    sys.exit(1)

# %% 4. Improvements (Data Augmentation and Dropout)
print("=== 4. Defining Data Augmentation and Model ===")

# Data augmentation layers
# Note: These run on the CPU by default unless placed inside the model
# and run on GPU during training. Placing them here is fine.
data_augmentation = keras.Sequential(
  [
    # Input shape is crucial for the first layer
    layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    # Add more augmentation if needed, e.g., RandomContrast
  ],
  name="data_augmentation",
)

# Create the Model
stop_event, thread = start_spinner(message=f"Building model for {num_classes} "
                                   "classes")
model = Sequential([
  # Apply data augmentation as the first layer
  data_augmentation,

  # Rescale pixel values from [0, 255] to [0, 1]
  layers.Rescaling(1./255),

  # Convolutional Block 1
  layers.Conv2D(16, kernel_size=3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  # Convolutional Block 2
  layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  # Convolutional Block 3
  layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  # Apply Dropout before the dense layers
  layers.Dropout(DROPOUT_RATE),

  # Flatten the feature maps into a vector
  layers.Flatten(),

  # Dense hidden layer
  layers.Dense(128, activation='relu'),

  # Output layer: num_classes units, no activation (logits output)
  layers.Dense(num_classes, name="outputs")
], name=f"{MODEL_TYPE}_classifier")

end_spinner(stop_event, thread)

# Compile the model
stop_event, thread = start_spinner(message="Compiling Model")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              # Use SparseCategoricalCrossentropy because labels are integers
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

end_spinner(stop_event, thread)

# Print model summary (optional)
# model.summary()

# %% 5. Train the Model
print(f"=== 5. Starting Training for {EPOCHS} Epochs ===")
try:
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=EPOCHS, 
      verbose=0
    )
    print("Training complete.")
except Exception as e:
    print(f"An error occurred during training: {e}")
    # Optionally exit or try to proceed depending on the error
    history = None # Ensure history is None if training failed

# %% 6. Visualize Training Results
print("=== 6. Visualizing Training Results ===")
if history: # Only plot if training was successful
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(EPOCHS)

    plt.figure(figsize=(6, 3)) # Wider figure to accommodate titles better

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", linewidth=1)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", linewidth=1)
    plt.legend(loc="lower right", fontsize=5)
    plt.title(f"{MODEL_TYPE.upper()} Training and Validation Accuracy", 
              fontsize=10)
    plt.xlabel("Epoch", fontsize=8)
    plt.ylabel("Accuracy", fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    # Dynamically set ylim based on data or keep fixed
    min_acc = min(min(acc), min(val_acc))
    plt.ylim([max(0, min_acc - 0.1), 1.05]) # Start below min accuracy, max 1.05
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", linewidth=1)
    plt.plot(epochs_range, val_loss, label="Validation Loss", linewidth=1)
    plt.legend(loc="upper right", fontsize=5)
    plt.title(f"{MODEL_TYPE.upper()} Training and Validation Loss", fontsize=10)
    plt.xlabel("Epoch", fontsize=8)
    plt.ylabel("Loss", fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    # Dynamically set ylim based on data
    max_loss = max(max(loss), max(val_loss))
    # Start below 0, go above max loss
    plt.ylim([-0.05, max(1.0, max_loss + 0.1)])
    
    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
else:
    print("Skipping visualization as training did not complete successfully.")

# %% 7. Predict on New Data
print("=== 7. Predicting on New Data ===")

# Check if the test image path exists before proceeding
if os.path.exists(test_image_path):
    print(f"Loading test image: {TEST_IMAGE_NAME}")

    # Visualize the test image using the imported function
    try:
        test_image_display_array = image_to_array(test_image_path)
        # Check if the array is suitable for imshow
        if isinstance(test_image_display_array, 
                      np.ndarray) and test_image_display_array.ndim == 3:
             plt.figure(figsize=(4, 4))
             ax = plt.gca()
             
             plt.imshow(test_image_display_array)
             ax.spines["left"].set_visible(False)
             ax.spines["bottom"].set_visible(False)
             ax.tick_params(left=False, bottom=False, labelleft=False, 
                            labelbottom=False)
             plt.title("Test Image", fontsize=9)
             plt.show()
        else:
            print("WARNING: Output of image_to_array is not a displayable "
                  "image array.")

    except Exception as e:
        print("Error processing or displaying test image with "
              f"image_to_array: {e}")

    # Prepare image for the model prediction
    try:
        img = tf.keras.utils.load_img(
            test_image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        # Make prediction
        predictions = model.predict(img_array)
        # Apply softmax to get probabilities because the model outputs logits
        score = tf.nn.softmax(predictions[0])

        predicted_class_index = np.argmax(score)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = 100 * np.max(score)
        
        print(
            "Prediction: This image most likely belongs to "
            f"'{predicted_class_name}' "
            f"with a {confidence:.2f}% confidence."
        )
        if "reservoir" in predicted_class_name:
            print("SUCCESS: model predicted correctly!")
        else:
            print("FAILURE: model predicted incorrectly")

    except FileNotFoundError:
        print(f"Error: Test image file not found at {TEST_IMAGE_NAME}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

else:
    print("Skipping prediction because test image was not found at: "
          f"{TEST_IMAGE_NAME}")

# %% 8. Save Model (Optional)
if SAVE_MODEL and history:
    print("=== 8. Saving Model ===")

    # Check if the primary path exists
    if os.path.exists(model_save_path):
        # Create a versioned filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(model_save_path)
        versioned_save_path = f"{base}_{timestamp}{ext}"
        print(f"WARNING: Original path {model_save_path} exists.")
        print(f"Attempting to save to versioned path: {versioned_save_path}")
        save_path_to_use = versioned_save_path
    else: # Use the original path if it doesn't exist
        save_path_to_use = model_save_path

    # Try saving to the determined path
    try:
        model.save(save_path_to_use)
        print(f"Model successfully saved to: {save_path_to_use}")
    except Exception as e:
        print(f"Error saving model: {e}")
elif not history:
     print("Skipping model saving as training did not complete successfully.")
else:
     print("Model saving is disabled.")


# %% 9. Final Summary
print("=== 9. Script End ===")
TOTAL_TIME = time.monotonic() - MAIN_START_TIME
print(f"Total processing time: {TOTAL_TIME:.2f} seconds")
print("================")
