import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import matplotlib.pyplot as plt

base_path = "img"

images = []
labels = []


print("Loading data...")
class_counts = {}
for label in os.listdir(base_path):
    folder_path = os.path.join(base_path, label)
    if not os.path.isdir(folder_path):  
        continue
    
    count = 0
    for img_file in os.listdir(folder_path):  
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))
        images.append(img)
        labels.append(ord(label.upper()) - ord('A'))
        count += 1
    
    class_counts[label.upper()] = count
    print(f"Class {label.upper()}: {count} images")


X = np.array(images, dtype=np.float32)
y = np.array(labels)

print(f"\nDataset Summary:")
print(f"Total images loaded: {len(images)}")
print(f"Image shape: {X[0].shape}")
print(f"Number of unique labels: {len(set(labels))}")
print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")


min_samples = min(class_counts.values())
max_samples = max(class_counts.values())
if max_samples / min_samples > 2:
    print(" Significant class imbalance detected!")
    print("Consider balancing your dataset or using class weights")


X /= 255.0


X = X[..., np.newaxis]


y = to_categorical(y, num_classes=4)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234, stratify=y.argmax(axis=1)
)

print(f"\nSplit sizes:")
print(f"Training: {X_train.shape[0]} samples")
print(f"Testing: {X_test.shape[0]} samples")

def create_simple_model():
    """Much simpler model to reduce overfitting"""
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(64, 64, 1)),
        
       
        keras.layers.Conv2D(16, (5, 5), padding='same'),  
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D((4, 4)),  
        keras.layers.Dropout(0.3),
        
     
        keras.layers.Conv2D(32, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D((4, 4)),  
        keras.layers.Dropout(0.4),
        
       
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        
       
        keras.layers.Dense(4, activation='softmax')
    ])
    
    model.summary()
    return model

def create_baseline_model():
    """Even simpler baseline model"""
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(64, 64, 1)),
        
       
        keras.layers.Conv2D(32, (7, 7), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D((8, 8)),  
        keras.layers.Dropout(0.5),
        
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(4, activation='softmax')
    ])
    
    print("Baseline model:")
    model.summary()
    return model

def train_with_heavy_regularization():
    
    model = create_simple_model()
    
    
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='nearest'
    )
    
    
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1234, 
        stratify=y_train.argmax(axis=1)
    )
    
    print(f"Final split sizes:")
    print(f"Training: {X_train_final.shape[0]}")
    print(f"Validation: {X_val.shape[0]}")
    
    
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_final.argmax(axis=1)),
        y=y_train_final.argmax(axis=1)
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005), 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.8,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("\nStarting training...")
    t0 = time.time()
    
    
    history = model.fit(
        datagen.flow(X_train_final, y_train_final, batch_size=16),  
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight_dict,  
        verbose=1
    )
    
    t1 = time.time()
    print(f"Training completed in {t1-t0:.2f} seconds")
    
    return model, history

def train_baseline():
    """Train ultra-simple baseline"""
    print("\n" + "="*50)
    print("TRAINING BASELINE MODEL")
    print("="*50)
    
    model = create_baseline_model()
    
    
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=1234,
        stratify=y_train.argmax(axis=1)
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),  
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    
    history = model.fit(
        X_train_final, y_train_final,
        epochs=20,
        validation_data=(X_val, y_val),
        batch_size=32,
        verbose=1
    )
    
    return model, history

print("Testing baseline model first...")
baseline_model, baseline_history = train_baseline()

baseline_test_loss, baseline_test_acc = baseline_model.evaluate(X_test, y_test, verbose=0)
print(f"\nBaseline Results:")
print(f"Test accuracy: {baseline_test_acc:.4f}")
print(f"Test loss: {baseline_test_loss:.4f}")

if baseline_test_acc > 0.3: 
    print("\nBaseline works! ")
    model, history = train_with_heavy_regularization()
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nComplex Model Results:")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0,0].plot(baseline_history.history['accuracy'], label='Train')
    axes[0,0].plot(baseline_history.history['val_accuracy'], label='Val')
    axes[0,0].set_title('Baseline Accuracy')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(baseline_history.history['loss'], label='Train')
    axes[0,1].plot(baseline_history.history['val_loss'], label='Val')
    axes[0,1].set_title('Baseline Loss')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(history.history['accuracy'], label='Train')
    axes[1,0].plot(history.history['val_accuracy'], label='Val')
    axes[1,0].set_title('Complex Model Accuracy')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(history.history['loss'], label='Train')
    axes[1,1].plot(history.history['val_loss'], label='Val')
    axes[1,1].set_title('Complex Model Loss')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
else:
    print(f"\n⚠️ BASELINE FAILED (acc={baseline_test_acc:.3f})")
    print("This suggests a fundamental data problem:")
    print("1. Check if your images are correctly labeled")
    print("2. Verify the classes are actually distinguishable")
    print("3. Make sure your folder structure is correct")
    print("4. Consider if you need more diverse training data")
    
    predictions = baseline_model.predict(X_test[:10], verbose=0)
    print("\nSample predictions vs actual:")
    for i in range(min(10, len(X_test))):
        pred_class = predictions[i].argmax()
        actual_class = y_test[i].argmax()
        confidence = predictions[i].max()
        print(f"Image {i}: Predicted={pred_class} (conf={confidence:.3f}), Actual={actual_class}")

print("\n" + "="*50)
print("DEBUGGING COMPLETE")
print("="*50)
model.save("models/asl_model7.keras")