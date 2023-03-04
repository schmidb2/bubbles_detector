import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import logging

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

    
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

folder_dir = "photos"

base_dir = os.path.join(os.path.dirname(folder_dir),'photos')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir,'validate')

train_bubbles_dir = os.path.join(train_dir,'bubbles')
train_non_bubbles_dir = os.path.join(train_dir,'non_bubbles')
validation_bubbles_dir = os.path.join(validation_dir,'bubbles')
validation_non_bubbles_dir = os.path.join(validation_dir,'non_bubbles')

num_bubs_tr = len(os.listdir(train_bubbles_dir))
num_non_bubs_tr = len(os.listdir(train_non_bubbles_dir))

num_bubs_val = len(os.listdir(validation_bubbles_dir))
num_non_bubs_val = len(os.listdir(validation_non_bubbles_dir))

total_train = num_bubs_tr + num_non_bubs_tr
total_val = num_bubs_val + num_non_bubs_val

print('total training bubbles images:', num_bubs_tr)
print('total training non bubbles images:', num_non_bubs_tr)
print('total validation bubbles images:', num_bubs_val)
print('total validation non bubbles images:', num_non_bubs_val)
print('--')
print('total training images:', total_train)
print('total validation images',total_val)

BATCH_SIZE = 50
IMG_SHAPE = 150

train_image_generator      = ImageDataGenerator(rescale=1./255,
                                                rotation_range=40,
                                                width_shift_range=0.2,
                                                height_shift_range=0.2,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True,
                                                fill_mode='nearest')

validation_image_generator = ImageDataGenerator(rescale=1./255) 

train_data_gen = train_image_generator.flow_from_directory(batch_size = BATCH_SIZE,
                                                           directory = train_dir,
                                                           shuffle = True,
                                                           target_size = (IMG_SHAPE,IMG_SHAPE),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size = (IMG_SHAPE,IMG_SHAPE),
                                                              class_mode='binary')

##sample_training_images, _ = next(train_data_gen)
##plotImages(sample_training_images[:5]) 


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

EPOCHS = 30
history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

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
plt.savefig('./foo.png')
plt.show()
##image_generator = ImageDataGenerator(rescaled=1./255)

##data = []
##folder_dir = "bubbles_photos"
##file_names = listdir(folder_dir)
##print(file_names)



##for images in os.listdir(folder_dir):
##    if(images.endswith(".jpg")):
##        
##        data.append(cv2.imread(images))
        
        
