import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # suppress info and warning messages
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import tensorflow.keras as keras

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
        img = cv2.resize(img, (300, 300))  #
        images.append(img)
        # 123
        labels.append(ord(label.upper()) - ord('A')) 

# Convert to NumPy arrays
X = np.array(images, dtype=np.float32)
y = np.array(labels)

# Normalize pixel values to [0, 1]
X /= 255.0

# Add channel dimension: (num_samples, 64, 64, 1)
X = X[..., np.newaxis]

# One-hot encode the labels: (num_samples, 26)
y = to_categorical(y, num_classes=26)  # 

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

def create_model():
    cnn_model = keras.Sequential()
    input_layer = keras.layers.InputLayer(input_shape = (300, 300, 1))
    cnn_model.add(input_layer)

    conv_1 = keras.layers.Conv2D(filters = 16, kernel_size = 3, padding ='same')
    batchNorm_1 = keras.layers.BatchNormalization()
    ReLU_1 = keras.layers.ReLU()
    pool_1 = keras.layers.MaxPooling2D((2, 2))
    dropout_1 = keras.layers.Dropout(0.25)
    cnn_model.add(conv_1)
    cnn_model.add(batchNorm_1)
    cnn_model.add(ReLU_1)
    cnn_model.add(dropout_1)
    cnn_model.add(pool_1)

    conv_2 = keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = "same")
    batchNorm_2 = keras.layers.BatchNormalization()
    ReLU_2 = keras.layers.ReLU()
    pool_2 = keras.layers.MaxPooling2D((2, 2))
    dropout_2 = keras.layers.Dropout(0.25)

    cnn_model.add(conv_2)
    cnn_model.add(batchNorm_2)
    cnn_model.add(ReLU_2)
    cnn_model.add(dropout_2)
    cnn_model.add(pool_2)

    conv_3 = keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same')
    batchNorm_3 = keras.layers.BatchNormalization()
    ReLU_3 = keras.layers.ReLU()
    pool_3 = keras.layers.MaxPooling2D((2,2))
    dropout_3 = keras.layers.Dropout(0.25)

    cnn_model.add(conv_3)
    cnn_model.add(batchNorm_3)
    cnn_model.add(ReLU_3)
    cnn_model.add(dropout_3)
    cnn_model.add(pool_3)



    conv_4 = keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same')
    batchNorm_4 = keras.layers.BatchNormalization()
    ReLU_4 = keras.layers.ReLU()
    pool_4 = keras.layers.MaxPooling2D((2,2))
    dropout_4 = keras.layers.Dropout(0.25)
    cnn_model.add(conv_4)
    cnn_model.add(batchNorm_4)
    cnn_model.add(ReLU_4)
    cnn_model.add(pool_4)
    cnn_model.add(dropout_4)


    pooling_layer = keras.layers.GlobalAveragePooling2D()
    cnn_model.add(pooling_layer)



    output_layer = keras.layers.Dense(units=26, activation = 'softmax')
    cnn_model.add(output_layer)
    cnn_model.summary()

    sgd_optimizer = keras.optimizers.SGD(learning_rate = 0.01)
    loss_fn = keras.losses.CategoricalCrossentropy()
    cnn_model.compile(optimizer = sgd_optimizer, loss = loss_fn, metrics = ['accuracy'])

    num_epochs = 20
    t0 = time.time()
    history = cnn_model.fit(X_train, y_train, epochs = num_epochs, validation_split = 0.2)
    t1 = time.time()
    
    return cnn_model, history


model, histoy = create_model()
test_loss, test_acc = model.evaluate(X_test, y_test)

print("test accuracy", test_acc)

model.save("models/asl_model4.h5")
