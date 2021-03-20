

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet import ResNet101
from keras.applications.resnet import ResNet152
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import inception_v3
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input as piss

import numpy as np

class TransferLearning():

    def __init__(self):
        self.augment_data = False
        self.base_model_type = "resnet50"
        self.img_width = 244
        self.img_height = 244
        self.batch_size = 64
        self.class_mode = 'categorical'
        self.shuffle = True
        self.seed = 666
        self.samples = 1000
        self.imageNetWeights = True

    # ################
    # Datagens
    # ################
    def get_vgg16_datagen(self, test=False):
        if(not test):
            vgg16_datagen = ImageDataGenerator(
                horizontal_flip=True,
                shear_range=0.4,
                zoom_range=0.4,
                data_format=str(keras.image_data_format()),
                preprocessing_function=keras.applications.vgg16.preprocess_input
            )
        else:
            vgg16_datagen = ImageDataGenerator(
                data_format=str(keras.image_data_format()),
                preprocessing_function=keras.applications.vgg16.preprocess_input
            )
        return vgg16_datagen

    def get_resnet50_datagen(self, test=False):
        if (not test):
            resnet50_datagen = ImageDataGenerator(
                horizontal_flip=True,
                shear_range=0.4,
                zoom_range=0.4,
                data_format=str(keras.image_data_format()),
                preprocessing_function=keras.applications.resnet50.preprocess_input
            )
        else:
            resnet50_datagen = ImageDataGenerator(
                data_format=str(keras.image_data_format()),
                preprocessing_function=keras.applications.resnet50.preprocess_input
            )
        return resnet50_datagen

    def get_resnet101_datagen(self, test=False):
        if(not test):
            resnet101_datagen = ImageDataGenerator(
                horizontal_flip=True,
                shear_range=0.4,
                zoom_range=0.4,
                data_format=str(keras.image_data_format()),
                preprocessing_function=keras.applications.resnet50.preprocess_input
            )
        else:
            resnet101_datagen = ImageDataGenerator(
                data_format=str(keras.image_data_format()),
                preprocessing_function=piss
            )
        return resnet101_datagen

    def get_resnet152_datagen(self, test=False):
        if(not test):
            resnet152_datagen = ImageDataGenerator(
                horizontal_flip=True,
                shear_range=0.4,
                zoom_range=0.4,
                data_format=str(keras.image_data_format()),
                preprocessing_function=keras.applications.resnet.preprocess_input
            )
        else:
            resnet152_datagen = ImageDataGenerator(
                data_format=str(keras.image_data_format()),
                preprocessing_function=keras.applications.resnet.preprocess_input
            )
        return resnet152_datagen

    def get_inception_v3_datagen(self, test_data=False):
        if(not test_data):
            inception_v3_train_datagen = ImageDataGenerator(
                horizontal_flip=True,
                shear_range=0.4,
                zoom_range=0.4,
                data_format=str(keras.image_data_format()),
                preprocessing_functwion=keras.applications.inception_v3.preprocess_input
            )
        else:
            inception_v3_test_datagen = ImageDataGenerator(
                data_format=str(keras.image_data_format()),
                preprocessing_function=keras.applications.inception_v3.preprocess_input
            )
        return inception_v3_train_datagen

    # ############
    # Input shape (general method)
    # ############
    def set_input_shape(self):
        if K.image_data_format() == 'channels_first':
            self.input_shape = (3, self.img_width, self.img_height)
            print("channels_first")
        else:
            self.input_shape = (self.img_width, self.img_height, 3)
            print("channels_last")

    # ##########
    # Base model
    # ##########
    def set_base_model(self):
        stringWeights = None
        if (self.imageNetWeights):
            print("Using imagenet weights...")
            stringWeights = "imagenet"
        else:
            print("Using random initialization of the weights...")
            stringWeights = None
        if (self.base_model_type == "vgg16"):
            self.base_model = keras.applications.vgg16.VGG16(include_top=False, weights=stringWeights, input_shape=self.input_shape,
                                                        pooling='max')
        elif (self.base_model_type == "resnet50"):
            self.base_model = ResNet50(include_top=False, weights=stringWeights, input_shape=self.input_shape, pooling='max')
        elif (self.base_model_type == "resnet101"):
            self.base_model = ResNet101(include_top=False, weights=stringWeights, input_shape=self.input_shape, pooling='max')
        elif (self.base_model_type == "resnet152"):
            self.base_model = ResNet152(include_top=False, weights=stringWeights, input_shape=self.input_shape, pooling='max')
        elif (self.base_model_type == "inceptionv3"):
            self.base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights=stringWeights,
                                                                     pooling='max')

    def make_layers_trainable(self, is_trainable = False, show_base_model_summary = False):
        for layer in self.base_model.layers:
            layer.trainable = is_trainable
        if (show_base_model_summary):
            self.base_model.summary()
            print(self.base_model.output_shape)

    # ############
    # Get datagen
    # ############
    def get_transfer_model_datagen(self, test_data = False, base_model="resnet50"):
        data_gen = self.get_vgg16_datagen()
        self.base_model_type = base_model
        if(self.base_model_type == "vgg16"):
            data_gen = self.get_vgg16_datagen(test_data=test_data)
        elif(self.base_model_type == "resnet50"):
            data_gen = self.get_resnet50_datagen(test_data=test_data)
        elif(self.base_model_type == "resnet101"):
            data_gen = self.resnet101_datagen(test_data=test_data)
        elif(self.base_model_type == "resnet152"):
            data_gen = self.resnet152_datagen(test_data=test_data)
        elif(self.base_model_type == "inceptionv3"):
            data_gen = self.inception_v3_datagen(test_data=test_data)
        print("base_model_predict_online:" ,self.base_model_type)
        return data_gen

    # ####################################
    # Using base models, preprocess data before feature extraction
    # ####################################
    def base_model_predict(self, generator, base_model):
      x_data, y_data = self.generate_data(generator) # generates x_data and y_data
      # Preprocessing the data, so that it can be fed to the pre-trained base_model.
      if(self.base_model_type == "vgg16"):
        print(" vgg16.preprocess_input(x_data)")
        _input = keras.applications.vgg16.preprocess_input(x_data)
      elif(self.base_model_type == "resnet50"):
        print(" resnet50.preprocess_input(x_data)")
        _input = keras.applications.resnet50.preprocess_input(x_data)
      elif(self.base_model_type == "resnet101"):
        print(" resnet101.preprocess_input(x_data)")
        _input = keras.applications.resnet.preprocess_input(x_data)
      elif(self.base_model_type == "resnet152"):
        print(" resnet152.preprocess_input(x_data)")
        _input = keras.applications.resnet.preprocess_input(x_data)
      elif(self.base_model_type == "inceptionv3"):
        print(" inception_v3.preprocess_input(x_data)")
        _input = keras.applications.inception_v3.preprocess_input(x_data)
      print("   base_model.predict(x_data)")
      features = base_model.predict(_input, verbose=1)
      return [features, x_data, y_data]

    # ###################################
    # Make online prediction with a generator (general)
    # returns the extracted features, the original data before feature extraction [x_data] and the labels [y_data]
    # ###################################
    def model_predict_online(self, model, data_gen, data_dir):
        features = model.predict_generator(data_gen.flow_from_directory(data_dir,
                                                                             target_size  = (self.img_width, self.img_height),
                                                                             batch_size = self.batch_size,
                                                                             class_mode = self.class_mode,
                                                                             shuffle = self.shuffle,
                                                                             seed = self.seed),
                                                steps = self.samples // self.batch_size,
                                                #          steps = (samples+extra_augmented_batches*batch_size+batch_size) // batch_size,
                                                verbose = 1)
        generator = data_gen.flow_from_directory(data_dir,
                                                 target_size  = (32, 32),
                                                 batch_size = self.batch_size,
                                                 class_mode = self.class_mode,
                                                 shuffle = self.shuffle,
                                                 seed = self.seed)

        x_data ,y_data = generator.next()
        counter = y_data.shape[0]
        while(True):
            x ,y = generator.next()
            x_data= np.concatenate((x_data, x), 0)
            y_data= np.concatenate((y_data, y), 0)
            counter =y_data.shape[0]
            if(len(features)//(counter + self.batch_size )==0):
                break
        return [features, x_data, y_data]