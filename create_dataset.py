import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import tensorflow.keras as keras
from tensorflow.keras import regularizers

import time
import matplotlib.pyplot as plt
import math


base_path = "img"

images = []
labels = []

#goes through the cata
for label in os.listdir(base_path):
    folder_path = os.path.join(base_path, label)
    if not os.path.isdir(folder_path):  
        continue
    for img_file in os.listdir(folder_path):  
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))  #
        images.append(img)
        # 123
        labels.append(ord(label.upper()) - ord('A')) 

# Convert to NumPy arrays
X = np.array(images, dtype=np.float32)
y = np.array(labels)
print(f"Total images loaded: {len(images)}")
print(f"Image shape: {X[0].shape}")
print(f"Number of unique labels: {len(set(labels))}")



# Normalize pixel values to [0, 1]
X /= 255.0

# Add channel dimension: (num_samples, 64, 64, 1)
X = X[..., np.newaxis]

# One-hot encode the labels: (num_samples, 26)
y = to_categorical(y, num_classes=4)  # 

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

def create_model():
    cnn_model = keras.Sequential()
    input_layer = keras.layers.InputLayer(input_shape = (64, 64, 1))
    cnn_model.add(input_layer)

    conv_1 = keras.layers.Conv2D(filters = 8, kernel_size = (3,3), padding ='same', kernel_regularizer = regularizers.l2(0.001))
    batchNorm_1 = keras.layers.BatchNormalization()
    ReLU_1 = keras.layers.ReLU()
    pool_1 = keras.layers.MaxPooling2D((2, 2))
    dropout_1 = keras.layers.Dropout(0.25)
    cnn_model.add(conv_1)
    cnn_model.add(batchNorm_1)
    cnn_model.add(ReLU_1)
    cnn_model.add(dropout_1)
    cnn_model.add(pool_1)

    conv_2 = keras.layers.Conv2D(filters = 16, kernel_size = 3, padding = "same")
    batchNorm_2 = keras.layers.BatchNormalization()
    ReLU_2 = keras.layers.ReLU()
    pool_2 = keras.layers.MaxPooling2D((2, 2))
    dropout_2 = keras.layers.Dropout(0.4)

    cnn_model.add(conv_2)
    cnn_model.add(batchNorm_2)
    cnn_model.add(ReLU_2)
    cnn_model.add(dropout_2)
    cnn_model.add(pool_2)

    conv_3 = keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same')
    batchNorm_3 = keras.layers.BatchNormalization()
    ReLU_3 = keras.layers.ReLU()
    pool_3 = keras.layers.MaxPooling2D((2,2))
    dropout_3 = keras.layers.Dropout(0.5)

    cnn_model.add(conv_3)
    cnn_model.add(batchNorm_3)
    cnn_model.add(ReLU_3)
    cnn_model.add(dropout_3)
    cnn_model.add(pool_3)


    pooling_layer = keras.layers.GlobalAveragePooling2D()
    cnn_model.add(pooling_layer)



    output_layer = keras.layers.Dense(units=4, activation = 'softmax')
    cnn_model.add(output_layer)
    cnn_model.summary()

    adam_optimizer = keras.optimizers.Adam(learning_rate = 0.001)
    loss_fn = keras.losses.CategoricalCrossentropy()
    cnn_model.compile(optimizer = adam_optimizer, loss = loss_fn, metrics = ['accuracy'])

    num_epochs = 20
    t0 = time.time()

    early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True, verbose = 1)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    history = cnn_model.fit(X_train, y_train, epochs = num_epochs, validation_split = 0.2, callbacks = [early_stopping, lr_scheduler])
    t1 = time.time()
    
    return cnn_model, history

"""
Good sign:
Training and validation accuracy go up together and level off.
Training and validation loss go down together and level off.

Overfitting sign:
Training accuracy keeps going up, but validation accuracy stops improving or drops.
Training loss keeps going down, but validation loss goes back up.
"""

model, history = create_model()
test_loss, test_acc = model.evaluate(X_test, y_test)

# Accuracy

plt.plot(history.history['accuracy'], label = 'Train Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# loss

plt.plot(history.history['loss'], label = 'Train Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


#model.save("models/asl_model6.keras")

