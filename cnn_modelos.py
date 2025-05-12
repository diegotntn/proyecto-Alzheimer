from tensorflow.keras import layers, Input, Model

def build_cnn1(input_shape):
    inputs = Input(shape=input_shape)

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    return Model(inputs, x, name='CNN1')

def build_cnn2(input_shape):
    inputs = Input(shape=input_shape)

    x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(512, (5, 5), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    return Model(inputs, x, name='CNN2')
