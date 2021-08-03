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
#definición de rutas
dirname = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + '\\data\\clases'
head_ct_url = 'No se ha seleccionado una imagen para predecir'

#Parametros de carga:
batch_size = 32
img_height = 180
img_width = 180
num_classes = 20
epochs=2

#Definicion de datos de entrenamiento
DS_training = tf.keras.preprocessing.image_dataset_from_directory(
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
DS_validation = tf.keras.preprocessing.image_dataset_from_directory(
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

#Definimos el número de clases
classes = DS_training.class_names



#Agregar AUTOTUNE al modelo, esto ayudará a que no se hagan cuellos  de botella
AUTOTUNE = tf.data.AUTOTUNE
DS_training = DS_training.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
DS_validation = DS_validation.cache().prefetch(buffer_size=AUTOTUNE)


#Se crea un layer de normalizacion para estandarizar datos
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

#Tecnicas de mejora de precisión
#Data augmentation: Medida en contra del overfitting(falta de imagenes de entrenamiento)
#Multiplica las imagenes y las entrena en otros angulos
KPreprocessing = layers.experimental.preprocessing
data_augmentation = keras.Sequential(
  [
    KPreprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    KPreprocessing.RandomFlip("horizontal_and_vertical", input_shape=(img_height, img_width, 3)),
    KPreprocessing.RandomRotation(0.1),
    KPreprocessing.RandomZoom(0.1),
    #KPreprocessing.RandomCrop(0.1, 0.1)
  ]
)


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

history = None

def getClasses():
  #Imprimir nombres de las clases
  return 'Clases: '+ '\n' + str(classes)
  

def getShape():
  #Imprimir shape de los modelos
  result = 'Estructura de los datos:\n'
  for image_batch, labels_batch in DS_training:
    result += str(image_batch.shape) + '\n'
    result += str(labels_batch.shape)
    break
  return result

def getDirectory():
  result = 'Carpeta de datos de entrenamiento:' + '\n'+dirname +'\n'
  result += 'Carpeta de datos de entrada:' +'\n'+ head_ct_url
  return result

def getBeggining():
  #return getClasses() +"\n" + getDirectory()+"\n" +getShape() +"\n" 
  return getClasses() +"\n" + getDirectory()+"\n"


def train():
  
  #Compilar el modelo
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])


  # Imprime el resumen de los datos del modelo
  model.summary()

  # Entrenamiento del modelo
  global history 
  history = model.fit(
    DS_training,
    validation_data=DS_validation,
    epochs=epochs
  )

  return history.history







def plot():
  # Impresión y creación de graficas para visualizar datos de entrenamiento
  if history is not None:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Certeza de entrenamiento',color='blue')
    plt.plot(epochs_range, val_acc, label='Certeza de validación',color='green')
    plt.legend(loc='lower right')
    plt.title('Certeza de entrenamiento y validación')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Perdida de entrenamiento',color='red')
    plt.plot(epochs_range, val_loss, label='Perdida de validación',color='orange')
    plt.legend(loc='upper right')
    plt.title('Función de perdida')
    plt.show()
  else:
    print('Not found')

def getEpochNumber():
  return epochs

def predict():

  img = keras.preprocessing.image.load_img(
      head_ct_url, target_size=(img_height, img_width)
  )
  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  result ="Esta imagen pertence al siguiente grupo \'{}\' con un porcentaje de confianza de {:.2f}.".format(classes[np.argmax(score)], 100 * np.max(score))
  print(result)

  return result

def setpredictingfile(predict_dir):
  global head_ct_url
  head_ct_url = predict_dir