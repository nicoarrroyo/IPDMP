""" Individual Project Keras Sequential Model (IPKSM)

Description:
This tutorial shows how to classify images of flowers using a 
`tf.keras.Sequential` model and load data using 
`tf.keras.utils.image_dataset_from_directory`. It demonstrates the following 
concepts:


* Efficiently loading a dataset off disk.
* Identifying overfitting and applying techniques to mitigate it, including 
data augmentation and dropout.

This tutorial follows a basic machine learning workflow:

1. Examine and understand data
2. Build an input pipeline
3. Build the model
4. Train the model
5. Test the model
6. Improve the model and repeat the process

In addition, the notebook demonstrates how to convert a [saved model] 
(../../../guide/saved_model.ipynb) to a [TensorFlow Lite] 
(https://www.tensorflow.org/lite/) model for on-device machine learning on 
mobile, embedded, and IoT devices.

Workflow:


Output:


"""
# %% Start
# %% Importing TensorFlow and Other Necessary Libraries
print("importing")
import time
MAIN_START_TIME = time.monotonic()
import pathlib
import os
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
print("importing complete")

try: # personal pc mode
    HOME = ("C:\\Users\\nicol\\OneDrive - " # personal computer user name
            "The University of Manchester\\Individual Project\\Downloads")
    os.chdir(HOME)
except: # uni mode
    HOME = ("C:\\Users\\c55626na\\OneDrive - " # university computer user name
            "The University of Manchester\\Individual Project\\Downloads")
    os.chdir(HOME)

epochs_keras_base = 5
epochs_keras_new = 5
model_type = "ndwi"

# %% Download and explore the dataset
"""

There are two datasets intended for the training of two models. Each is made 
up of about 1,300 photos of either water reservoirs, or non-reservoir water 
bodies. The two datasets exist to train one model on Normalised Difference 
Water Index (NDWI) images, while the other will be trained on True Colour 
Images (TCI). In each of these folders, there will be two sub-directories, one 
per class:

```
ndwi/
  reservoirs/
  water bodies/
```
```
tci/
  reservoirs/
  water bodies/
```

"""
folder = "S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU_20250301T152054.SAFE"
path = f"{HOME}\\Sentinel 2\\{folder}\\data"
data_dir = path # Set data directory to a subdirectory

# Extract if the extracted directory doesn't exist
extracted_dir = os.path.join(data_dir, f"{model_type}")

data_dir = pathlib.Path(extracted_dir)
image_count = len(list(data_dir.glob('*/*.png'))) # find all pngs in data_dir
print(f"image count: {image_count}")

# %% Load data using a Keras utility
"""

Next, load these images off disk using the helpful 
`tf.keras.utils.image_dataset_from_directory` utility. This will take you from 
a directory of images on disk to a `tf.data.Dataset` in just a couple lines of 
code. If you like, you can also write your own data loading code from scratch 
by visiting the [Load and preprocess images](../load_data/images.ipynb) 
tutorial.

Define some parameters for the loader:
"""

# %%% Create a dataset

batch_size = 32
img_height = 157
img_width = 157
validation_split = 0.2

"""It's good practice to use a validation split when developing your model. 
Use 80% of the images for training and 20% for validation."""

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=validation_split,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=validation_split,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# %% Visualize the data
"""

Here are the first nine images from the training dataset:
"""
class_names = train_ds.class_names
plt.figure(figsize=(6, 6))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]], fontsize=6)
    plt.axis("off")

"""You will pass these datasets to the Keras `Model.fit` method for training 
later in this tutorial. If you like, you can also manually iterate over the 
dataset and retrieve batches of images."""

"""The `image_batch` is a tensor of the shape `(32, 180, 180, 3)`. This is a 
batch of 32 images of shape `180x180x3` (the last dimension refers to color 
channels RGB). The `label_batch` is a tensor of the shape `(32,)`, these are 
corresponding labels to the 32 images.

You can call `.numpy()` on the `image_batch` and `labels_batch` tensors to 
convert them to a `numpy.ndarray`.

"""

# %% Configure the dataset for performance
"""

Make sure to use buffered prefetching, so you can yield data from disk without 
having I/O become blocking. These are two important methods you should use 
when loading data:

- `Dataset.cache` keeps the images in memory after they're loaded off disk 
during the first epoch. This will ensure the dataset does not become a 
bottleneck while training your model. If your dataset is too large to fit into 
memory, you can also use this method to create a performant on-disk cache.
- `Dataset.prefetch` overlaps data preprocessing and model execution while 
training.

Interested readers can learn more about both methods, as well as how to cache 
data to disk in the *Prefetching* section of the [Better performance with the 
tf.data API](../../guide/data_performance.ipynb) guide.
"""

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %% Standardize the data
"""

The RGB channel values are in the `[0, 255]` range. This is not ideal for a 
neural network; in general you should seek to make your input values small.

Here, you will standardize values to be in the `[0, 1]` range by using 
`tf.keras.layers.Rescaling`:
"""

normalization_layer = layers.Rescaling(1./255)

"""There are two ways to use this layer. You can apply it to the dataset by 
calling `Dataset.map`:"""

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

"""Or, you can include the layer inside your model definition, which can 
simplify deployment. Use the second approach here.

Note: You previously resized images using the `image_size` argument of 
`tf.keras.utils.image_dataset_from_directory`. If you want to include the 
resizing logic in your model as well, you can use the 
`tf.keras.layers.Resizing` layer.
"""
# %% A basic Keras model
"""
The Keras [Sequential](https://www.tensorflow.org/guide/keras/sequential_model) 
model consists of three convolution blocks (`tf.keras.layers.Conv2D`) with a 
max pooling layer (`tf.keras.layers.MaxPooling2D`) in each of them. There's a 
fully-connected layer (`tf.keras.layers.Dense`) with 128 units on top of it 
that is activated by a ReLU activation function (`'relu'`). This model has not 
been tuned for high accuracy; the goal of this tutorial is to show a standard 
approach.
"""
# %%% Create the model

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# %%% Compile the model
"""

For this tutorial, choose the `tf.keras.optimizers.Adam` optimizer and 
`tf.keras.losses.SparseCategoricalCrossentropy` loss function. To view 
training and validation accuracy for each training epoch, pass the `metrics` 
argument to `Model.compile`.
"""

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# %%% Model summary
"""

View all the layers of the network using the Keras `Model.summary` method:
"""

model.summary()

# %%% Train the model
"""

Train the model for 10 epochs with the Keras `Model.fit` method:
"""

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs_keras_base
)

# %% Visualize training results
"""

Create plots of the loss and accuracy on the training and validation sets:
"""

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs_keras_base)

plt.figure(figsize=(6, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy', fontsize=9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss', fontsize=9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.show()

"""The plots show that training accuracy and validation accuracy are off by 
large margins, and the model has achieved only around 60% accuracy on the 
validation set.

The following tutorial sections show how to inspect what went wrong and try to 
increase the overall performance of the model.

"""

# %% Improvements
# %%% Overfitting

"""
In the plots above, the training accuracy is increasing linearly over time, 
whereas validation accuracy stalls around 60% in the training process. Also, 
the difference in accuracy between training and validation accuracy is 
noticeable—a sign of [overfitting] 
(https://www.tensorflow.org/tutorials/keras/overfit_and_underfit).

When there are a small number of training examples, the model sometimes learns 
from noises or unwanted details from training examples—to an extent that it 
negatively impacts the performance of the model on new examples. This 
phenomenon is known as overfitting. It means that the model will have a 
difficult time generalizing on a new dataset.

There are multiple ways to fight overfitting in the training process. In this 
tutorial, you'll use *data augmentation* and add *dropout* to your model.
"""
# %%% Data augmentation

"""
Overfitting generally occurs when there are a small number of training 
examples. [Data augmentation](./data_augmentation.ipynb) takes the approach of 
generating additional training data from your existing examples by augmenting 
them using random transformations that yield believable-looking images. This 
helps expose the model to more aspects of the data and generalize better.

You will implement data augmentation using the following Keras preprocessing 
layers: `tf.keras.layers.RandomFlip`, `tf.keras.layers.RandomRotation`, and 
`tf.keras.layers.RandomZoom`. These can be included inside your model like 
other layers, and run on the GPU.
"""

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

"""Visualize a few augmented examples by applying data augmentation to the 
same image several times:"""

plt.figure(figsize=(6, 6))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
print("this is what data augmentation looks like", flush=True)
plt.show()
"""You will add data augmentation to your model before training in the next 
step."""
# %%% Dropout

"""
Another technique to reduce overfitting is to introduce [dropout] 
(https://developers.google.com/machine-learning/glossary#dropout_regularization) 
 regularization to the network.

When you apply dropout to a layer, it randomly drops out (by setting the 
activation to zero) a number of output units from the layer during the 
training process. Dropout takes a fractional number as its input value, in the 
form such as 0.1, 0.2, 0.4, etc. This means dropping out 10%, 20% or 40% of 
the output units randomly from the applied layer.

Create a new neural network with `tf.keras.layers.Dropout` before training it 
using the augmented images:
"""

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])
# %%% Compile and train the model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs_keras_new
)

# %%% Visualize training results
"""

After applying data augmentation and `tf.keras.layers.Dropout`, there is less 
overfitting than before, and training and validation accuracy are closer 
aligned:
"""

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs_keras_new)

plt.figure(figsize=(6, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy', fontsize=9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss', fontsize=9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.show()

# %%% Predict on new data
"""

Use your model to classify an image that wasn't included in the training or 
validation sets.

Note: Data augmentation and dropout layers are inactive at inference time.
"""

test_image_name = "ndwi chunk 1 reservoir 1.png"
test_image_path = f"{path}\\ndwi\\{test_image_name}"

img = tf.keras.utils.load_img(
    test_image_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# %% Final
TOTAL_TIME = time.monotonic() - MAIN_START_TIME
# =============================================================================
# model.save("C:\\Users\\nicol\\Documents\\UoM\\YEAR 3\\Individual Project"
#            "\\IPMLS\\IPRFM testing\\tci_model_1.keras")
# =============================================================================
print(f"total processing time: {round(TOTAL_TIME, 2)} seconds", flush=True)
