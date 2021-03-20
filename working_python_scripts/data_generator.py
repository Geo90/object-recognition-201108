
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
import keras
from datagen import DataGen

# 1. create a data_gen
# 2. use data_gen in generator
# 3. generate data [x,y]

class DataGenerator():

    def __init__(self, dir='Images', augment_data = True, img_width=32, img_height=32, batch_size=64, class_mode='categorical', shuffle=True, seed=666):
        self.dir = dir
        self.augment_data = augment_data
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.class_mode = class_mode
        self.shuffle = shuffle
        self.seed = seed
        # self.nb_train_samples = sum([len(files) for r, d, files in os.walk(dir+"/train")])
        # self.nb_validation_samples = sum([len(files) for r, d, files in os.walk(dir+"validation")])

    # #############################################
    # Load data from directory and return generator (general)
    # #############################################
    def load_data_from_directory(self, data_gen):
        generator = data_gen.flow_from_directory(
            self.dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=self.shuffle,
            seed=self.seed
        )
        return generator


    # #######################
    # Used for getting the data generators, if there is a train and validation dir
    # #######################
    def load_data_from_directory(self, data_gen, dir='train'):
        if dir == 'train':
            generator = data_gen.flow_from_directory(
                dir,
                target_size = (self.img_width, self.img_height),
                batch_size = self.batch_size,
                class_mode = 'categorical',
                shuffle = self.shuffle,
                seed = self.seed
                )
        else:
            generator = data_gen.flow_from_directory(
                  dir,
                  target_size=(self.img_width, self.img_height),
                  batch_size=self.batch_size,
                  class_mode='categorical',
                  shuffle=self.shuffle,
                  seed=self.seed
                  )
        return generator

    # ############################
    # Generate data from the generators from load_data_from_directory (general)
    # ############################
    def generate_data(self, generator, train_data = True):
        print("generate_data(generator)")
        self.samples = self.nb_train_samples
        if(not train_data):
            self.samples = self.nb_validation_samples
        generator.reset()
        x_data, y_data = generator.next()
        counter = x_data.shape[0]
        print(counter)
        print(self.samples)
        while (counter + self.batch_size <= self.samples):
            x, y = generator.next()
            x_data = np.concatenate((x_data, x), 0)
            y_data = np.concatenate((y_data, y), 0)
            counter += x.shape[0]
            print("{:.2f}%".format(counter / self.samples * 100))
        return x_data, y_data

    def test_data_generator(self, dir='test'):
        test_datagen = ImageDataGenerator(
              data_format=str(keras.backend.image_data_format()))
        validation_generator = test_datagen.flow_from_directory(
                  dir,
                  target_size = (self.img_width, self.img_height),
                  batch_size = self.batch_size,
                  class_mode = 'categorical',
                  shuffle = self.shuffle,
                  seed = self.seed
                  )
        return validation_generator

    # Create a dataset. Obtaining a labeled dataset from image files on disk
    def create_dataset(self, path_to_main_directory='Images'):
        dataset = keras.preprocessing.image_dataset_from_directory(
          path_to_main_directory, batch_size=64, image_size=(64, 64))
        print("create dataset")
        # For demonstration, iterate over the batches yielded by the dataset.
        for data, labels in dataset:
           print(data.shape)  # (64, 200, 200, 3)
           print(labels.shape)  # (64,)


    # obtaining a labeled dataset from text files on disk
    def create_text_dataset(self, path_to_main_directory='Texts'):
        dataset = keras.preprocessing.text_dataset_from_directory(
            'path/to/main_directory', batch_size=64)

        # For demonstration, iterate over the batches yielded by the dataset.
        for data, labels in dataset:
            print(data.shape)  # (64,)
            print(data.dtype)  # string
            print(labels.shape)  # (64,)
            print(labels.dtype)  # int32
        return dataset

dg = DataGenerator()
dataset = dg.create_dataset()
print(dataset)