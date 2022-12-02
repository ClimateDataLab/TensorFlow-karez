import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, preprocessing, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, BatchNormalization
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.backend import backend as K
from tensorflow.python.keras.backend import eval
from tensorflow.python.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.callbacks import TensorBoard
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import csv
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as img

#---parameters
l2_param                = 0.1#0.001 # L2 regularisation to prevent overfit 0-0.1
momentum_param          = 0.9  # default 0.99 reducing it to dampenin oscillations
lr_param                = 0.0001# reduced from default 0.01
batch_size              = 8    # default
num_of_valid_samples    = 112#740
num_of_train_samples    = 48#2960
imx                     = 198
imy                     = 198

#dirPath='C:/sarah/kfold/buffer100_scale/'
dirPath='C:/sarah/kfold/img_from_tile/'

#--- set up the model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(imx, imy, 1),padding='valid',kernel_regularizer=tf.keras.regularizers.l2(l2_param)))
model.add(Activation('relu'))
model.add(BatchNormalization(momentum=momentum_param))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(l2_param)))
model.add(Activation('relu'))
model.add(BatchNormalization(momentum=momentum_param))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(l2_param)))
model.add(Activation('relu'))
model.add(BatchNormalization(momentum=momentum_param))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#--- set up TensorBoard
#logdir = "C:\\sarah\\kfold\\logs\\lr_1e-4_adam_momentum_0pt9_batch_32_nofold_big_l2_0pt1" + datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "C:\\sarah\\kfold\\logs\\lr_1e-4_adam_momentum_0pt9_batch_32_nofold_big_l2_0pt1_ba" + datetime.now().strftime("%Y%m%d-%H%M%S")
myTensorBoard=TensorBoard(log_dir=logdir)

# --- in terminal type this to get the tensorBoard url
# --- tensorboard --logdir="C:\\sarah\\kfold\\logs\\" --bind_all

model.compile(tf.keras.optimizers.Adam(lr=lr_param), loss='binary_crossentropy', metrics=['binary_accuracy']) # used in hot dog
#model.compile(tf.keras.optimizers.Adam(lr=lr_param), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()]) #area under curve metric

# only augment traning data
train_datagen = ImageDataGenerator(
                rotation_range=180,
                width_shift_range=0.1,
                height_shift_range=0.1,
                rescale=1./255,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip='True',
                fill_mode='nearest')

# this is a generator that will read pictures found in subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        dirPath + "/train",
        target_size=(imx, imy),
        batch_size=batch_size,
        color_mode='grayscale',
        shuffle='True',
        seed=42,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

#--- no need to augment just rescale
valid_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = valid_datagen.flow_from_directory(
        dirPath + '/valid/',
        target_size=(imx, imy),
        batch_size=batch_size,
        color_mode='grayscale',
        shuffle=False,
        seed=42,
        class_mode='binary')

#valid_generator.reset()

#--- 20 epochs is good
history = model.fit_generator(
        train_generator,
        epochs=50,
        shuffle='True',
        steps_per_epoch=10,
        validation_steps=10,
        validation_data=validation_generator,
        callbacks=[myTensorBoard])

#history = model.fit_generator(
 #       train_generator,
  #      epochs=50,
   #     shuffle='True',
    #    steps_per_epoch=num_of_train_samples // batch_size,
     #   validation_steps=num_of_valid_samples // batch_size,
      #  validation_data=validation_generator,
       # callbacks=[myTensorBoard])

print(history.history.keys())

model.summary()
model.save('karez_cnn_buffer100_scale.h5')

#Confution Matrix and Classification Report
#val_predicts = model.predict_generator(validation_generator,num_of_valid_samples // batch_size+1)
val_predicts = model.predict_generator(validation_generator,14)

#---round up/down so predictions are 0 or 1. Needed for a correct confusion matrix.
y_pred = 1*(val_predicts >= 0.5)

f = open("today.txt",'w')
for i in range(num_of_valid_samples):
        f.write(str(y_pred[i]) + validation_generator.filenames[i])
        f.write('\n')

f.close()


#print(y_pred)

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['karez', 'no karez']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

#train_acc = history.history['auc']
#train_loss = history.history['loss']

#valid_acc = history.history['val_auc']
#valid_loss = history.history['val_loss']



# --- make a predictions on the test data
#test_generator.reset()
#predictions = model.predict_generator(test_generator)
#print(predictions)
#score = predictions[0]
#for x in range(0, 3):

 #   print(
 #   "This image is %.2f percent karez and %.2f percent not karez."
 #   % (100 * (1 - predictions[x]), 100 * predictions[x])
#)


#plt.subplot(2,1,1)
#plt.plot(train_loss, 'g', label='Training loss')
#plt.plot(valid_loss, 'b', label='Validation loss')
#plt.title('Training and Validation loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.subplot(2,1,2)
#plt.plot(train_acc, 'g', label='Training acc')
#plt.plot(valid_acc, 'b', label='Validation acc')
#plt.title('Training and Validation acc')
#plt.xlabel('Epochs')
#plt.ylabel('acc')
#plt.legend()
#plt.show(block=True)
#plt.interactive(False)

#--- now train the model
#history = model.fit_generator(
#        train_generator,
#        steps_per_epoch=2000 // batch_size,
#        epochs=50,
#        validation_data=validation_generator,
#        validation_steps=800 // batch_size) #ignoring this means all the valid data is used

#--- no need to augment just rescale
#test_datagen = ImageDataGenerator(rescale=1./255)

#--- run predictions on the test data
#test_generator = test_datagen.flow_from_directory(
#        testDir,
#        target_size=(99, 99),
#        batch_size=batch_size,
#        color_mode='grayscale',
#        class_mode='binary')

#--- plot training and valid images
#im = img.imread('//dirpath/ml_karez/train/karez/shaft_011.png')
#plt.subplot(2,1,1)
#plt.imshow(im)
#im = img.imread('//dirpath/ml_karez/train/nokarez/shaft_011.png')
#plt.subplot(2,1,2)
#plt.imshow(im)
#plt.show(block=True)
#plt.interactive(False)
