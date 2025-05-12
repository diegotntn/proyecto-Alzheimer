from tensorflow.python.client import device_lib
import tensorflow as tf

print("\n📟 Dispositivos disponibles:")
print(device_lib.list_local_devices())

# También para saber si TensorFlow detecta una GPU:
print("\n🚀 TensorFlow detecta GPU:", tf.config.list_physical_devices('GPU'))
