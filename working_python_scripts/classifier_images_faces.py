
import keras
import cv2
import numpy as np
from keras import Model
from keras.callbacks import *
import matplotlib.pyplot as plt
import os
from data_generator import DataGenerator
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def normalize_image(img):
    norm_img = np.zeros((300, 300))
    norm_img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    return norm_img

def show_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)

# Neural networks need all input images to have the same shape and size
# img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
# show_img(img)

dg = DataGenerator
train_generator, validation_generator = dg.load_data_from_directory("Images/train", "Images/validation")
train_data, train_target = dg.generate_data(train_generator)
val_data, val_target = dg.generate_data(validation_generator)

epochs = 15

num_classes = 2
inputs = keras.layers.Input(shape=train_data.shape[1:], name = "Input")

x = keras.layers.BatchNormalization()(inputs)
x = keras.layers.Dense(4, activation='relu', name="dense1")(x)
x = keras.layers.Conv2D(32, (6,6), padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D(pool_size=(4,4))(x)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(4, activation='relu', name="dense2")(x)

prediction = keras.layers.Dense(num_classes, activation='softmax')(x)
model = Model(inputs=inputs, outputs=prediction, name="prediction")
model.compile(loss="categorical_crossentropy", optimizer='SGD', metrics=['acc'])

print(model.summary())
print("##########")
print(train_data.shape)
print(train_target.shape)
print(val_data.shape)
print(val_target.shape)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
callbacks_list = [es]
history = model.fit(train_data, train_target, epochs=epochs,
                    callbacks=callbacks_list, validation_data=(val_data, val_target))

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# Save figure
# filepath = str(saveString) + "_acc.jpg"
# plt.savefig(filepath)
plt.show()

model.save("model_architecture.hd5")
model.save_weights("model_weights.hd5")
