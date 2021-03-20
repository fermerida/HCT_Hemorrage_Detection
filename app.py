import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


#Se ingresa la direccion que contiene las clases de clasificacion
#Se considera una carpeta para cada clase con  imagenes que la representan dentro de ella
dirname = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + '\\data\\clases'
print(dirname)

#Parametros de carga:
batch_size = 32
img_height = 180
img_width = 180
num_classes = 2
epochs=15

#Definicion de datos de entrenamiento
training = tf.keras.preprocessing.image_dataset_from_directory(
    dirname, 
    labels='inferred', 
    label_mode='int',
    class_names=None, 
    color_mode='rgb', 
    batch_size=batch_size, 
    image_size=(img_height, img_width),
    seed=123, 
    validation_split=0.2, 
    subset='training',
    interpolation='bilinear', 
    follow_links=False
)

#Definicion de datos de validación
validation = tf.keras.preprocessing.image_dataset_from_directory(
    dirname, 
    labels='inferred', 
    label_mode='int',
    class_names=None, 
    color_mode='rgb', 
    batch_size=batch_size, 
    image_size=(img_height, img_width),
    seed=123, 
    validation_split=0.2, 
    subset='validation',
    interpolation='bilinear', 
    follow_links=False
)

#Imprimir nombres de las clases
classes = training.class_names
print(classes)

#Imprimir shape de los modelos
for image_batch, labels_batch in training:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


#Agregar AUTOTUNE al modelo, esto ayudará a que no se hagan cuellos  de botella
AUTOTUNE = tf.data.AUTOTUNE
training = training.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = training.cache().prefetch(buffer_size=AUTOTUNE)


#Se crea un layer de normalizacion para estandarizar datos
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

#Tecnicas de mejora de precisión
#Data augmentation: Medida en contra del overfitting(falta de imagenes de entrenamiento)
#Multiplica las imagenes y las entrena en otros angulos

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

#Dropout: 



#Creación de modelo
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


#Compilar el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# Imprime el resumen de los datos del modelo
model.summary()

# Entrenamiento del modelo
history = model.fit(
  training,
  validation=val_ds,
  epochs=epochs
)




# Impresión y creación de graficas para visualizar datos de entrenamiento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



head_ct_url = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + '\\data\\analyze\\099.png'
print(head_ct_url)

img = keras.preprocessing.image.load_img(
    head_ct_url, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "Esta imagen pertence al siguiente grupo \'{}\' con un porcentaje de confianza de {:.2f}."
    .format(classes[np.argmax(score)], 100 * np.max(score))
)
