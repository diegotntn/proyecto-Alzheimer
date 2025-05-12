import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Activar Mixed Precision (requiere GPU moderna)
mixed_precision.set_global_policy('mixed_float16')

# =======================
# Parámetros
# =======================
input_shape = (224, 224, 3)
num_classes = 4
class_names = ['AD', 'CN', 'EMCI', 'LMCI']
batch_size = 32
epochs = 10

# =======================
# Modelo combinado con entrada única
# =======================
def build_combined_model_single_input(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # CNN 1
    x1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.5)(x1)
    x1 = layers.Flatten()(x1)
    x1 = layers.Dense(128, activation='relu')(x1)

    # CNN 2
    x2 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    x2 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = layers.Conv2D(128, (5, 5), activation='relu', padding='same')(x2)
    x2 = layers.Conv2D(128, (5, 5), activation='relu', padding='same')(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = layers.Conv2D(512, (5, 5), activation='relu', padding='same')(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.5)(x2)
    x2 = layers.Flatten()(x2)
    x2 = layers.Dense(128, activation='relu')(x2)

    combined = layers.concatenate([x1, x2])
    output = layers.Dense(num_classes, activation='softmax', dtype='float32')(combined)

    return Model(inputs=inputs, outputs=output)

model = build_combined_model_single_input(input_shape, num_classes)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =======================
# Carga de datos
# =======================
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# =======================
# Entrenamiento
# =======================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

# =======================
# Guardar modelo
# =======================
model.save("dual_cnn_model.h5")
print("\n✅ Modelo guardado como 'dual_cnn_model.h5'")

# =======================
# Evaluación y métricas
# =======================
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Visualización de predicciones
data_iter = iter(test_gen)
for i in range(5):
    img, label = next(data_iter)
    pred = model.predict(img)
    pred_class = class_names[np.argmax(pred)]
    true_class = class_names[np.argmax(label)]

    plt.imshow(img[0])
    plt.title(f"Real: {true_class} | Predicho: {pred_class}")
    plt.axis('off')
    plt.show()
