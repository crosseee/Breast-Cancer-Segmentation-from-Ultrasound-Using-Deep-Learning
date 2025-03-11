import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from keras.callbacks import EarlyStopping

path = 'D:/Downloads/Study/deep_learning/dataset/archive/Dataset_BUSI_with_GT/'
data_dir = pathlib.Path(path)

class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
class_names

batch_size = 16
img_height = 224
img_width = 224

from keras.utils import image_dataset_from_directory

train_data = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=233,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_data = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=250,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

from keras import layers
model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="Adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

epochs = 10

early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=3,  
    restore_best_weights=True 
)

history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data,
    batch_size=batch_size,
    callbacks=[early_stopping] 
)

history.history.keys()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Accuracy')
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Loss')
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend()
plt.show()

model.evaluate(val_data)

model.summary()
