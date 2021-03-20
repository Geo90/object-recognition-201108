
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

class DataGen():

    def __init__(self, augment_data=False):
        self.augment_data = augment_data

    def data_gen(self):
        # augmentation configuration, used with train_generator in train_datagen.flow()
        if (self.augment_data):
            data_gen = ImageDataGenerator(
                horizontal_flip=True,
                shear_range=0.4,
                zoom_range=0.4,
                data_format=str(K.image_data_format()),
            )
        else:
            data_gen = ImageDataGenerator(
                data_format=str(K.image_data_format())
            )
        return data_gen
