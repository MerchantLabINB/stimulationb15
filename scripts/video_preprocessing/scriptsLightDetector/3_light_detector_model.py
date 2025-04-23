import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directorios de las imágenes
light_frames_dir = 'light_frames'
no_light_frames_dir = 'no_light_frames'

# Cargar las imágenes y etiquetas
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(label)
    return images, labels

# Cargar las imágenes
light_images, light_labels = load_images_from_folder(light_frames_dir, 1)
no_light_images, no_light_labels = load_images_from_folder(no_light_frames_dir, 0)

# Combinar las imágenes y etiquetas
all_images = np.array(light_images + no_light_images)
all_labels = np.array(light_labels + no_light_labels)

# Normalizar las imágenes
all_images = all_images / 255.0  # Normalización a rango [0, 1]

# Añadir una dimensión para canales (TensorFlow espera [batch_size, height, width, channels])
all_images = np.expand_dims(all_images, axis=-1)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Definir generadores de augmentación de imágenes
# Augmentation normal para la clase mayoritaria (light)
datagen_majority = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Augmentation más intensiva para la clase minoritaria (no_light)
datagen_minority = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Aplicar data augmentation solo en el conjunto de entrenamiento
datagen_majority.fit(X_train[y_train == 1])
datagen_minority.fit(X_train[y_train == 0])

# Crear generadores de datos
train_majority_gen = datagen_majority.flow(X_train[y_train == 1], y_train[y_train == 1], batch_size=8, shuffle=True)
train_minority_gen = datagen_minority.flow(X_train[y_train == 0], y_train[y_train == 0], batch_size=8, shuffle=True)

# Combinar ambos generadores para entrenamiento
def combined_generator(gen1, gen2):
    while True:
        X1, y1 = gen1.next()
        X2, y2 = gen2.next()
        yield np.concatenate((X1, X2)), np.concatenate((y1, y2))

train_gen = combined_generator(train_majority_gen, train_minority_gen)

# Calcular las ponderaciones de las clases para el desbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Definir el modelo en TensorFlow con capas adicionales y batch normalization
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(all_images.shape[1], all_images.shape[2], 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Dropout layer added for regularization
    
    tf.keras.layers.Dense(1, activation='sigmoid')  # Activación sigmoide para clasificación binaria
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping para evitar sobreajuste
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Calcular los pasos por época
steps_per_epoch = min(len(train_majority_gen), len(train_minority_gen))

# Entrenar el modelo con augmentación y ponderaciones de clase
history = model.fit(train_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=50, validation_data=(X_test, y_test),
                    callbacks=[early_stopping], class_weight=class_weights)

# Guardar el modelo
model.save('redlight_detector_model_augmented_2808.h5')

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nPrecisión en el conjunto de prueba: {test_acc:.4f}')

# Visualizar la curva de aprendizaje
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()
