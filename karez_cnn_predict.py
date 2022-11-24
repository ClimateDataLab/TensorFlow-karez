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

batch_size              = 16
num_of_valid_samples    = 6351

imx                     = 198
imy                     = 198

dirPath='C:/sarah/test_gdal_retile/'
model = tf.keras.models.load_model('karez_cnn_buffer100_scale.h5')

# Check its architecture
model.summary()

valid_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = valid_datagen.flow_from_directory(
        dirPath,
        target_size=(imx, imy),
        batch_size=batch_size,
        color_mode='grayscale',
        shuffle=True,
        seed=42,
        class_mode='binary')

#Confution Matrix and Classification Report
val_predicts = model.predict_generator(validation_generator,num_of_valid_samples // batch_size+1)


y_pred = 1*(val_predicts >= 0.5)

#---write yes,no and filename to textfile
f = open("predict_today.txt",'w')

for i in range(num_of_valid_samples):
        f.write(str(y_pred[i]) + validation_generator.filenames[i])
        f.write('\n')

f.close()

#print('Confusion Matrix')
#print(confusion_matrix(validation_generator.classes, y_pred))
#print('Classification Report')
#target_names = ['karez', 'no karez']
#print(classification_report(validation_generator.classes, y_pred, target_names=target_names))




#predicted_class_indices=np.argmax(val_predicts,axis=1)
#labels=(validation_generator.class_indices)
#labels2=dict((v,k) for k,v in labels.items())
#predictions=[labels2[k] for k in predicted_class_indices]
#print(predicted_class_indices)
#print(labels)
#print(predictions)