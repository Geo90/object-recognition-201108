import numpy as np
import matplotlib.pyplot as plt
import sys
from keras.models import load_model
from data_generator import DataGenerator

class webcam_predictor():

    def __init__(self, model_path="model_architecture.hd5", weights_path="model_weights.hd5"):
        print("webcam_predictor __init__")
        self.model = load_model(model_path)
        self.model.load_weights(weights_path)

        print(self.model.summary())
        self.PERSON_LIST = ["Dominika", "George"]

        self.img_width = self.img_height = 64
        self.test_dir = "Images/test"
        self.batch_size = 32
        self.shuffle = True
        self.seed = 666

    def load(self, img):
       np_image = np.array(img).astype('float32')
       np_image = np.expand_dims(np_image, axis=0)
       return np_image


    def predict(self, img):
        y_pred=self.model.predict(self.load(img))
        print(y_pred[0])
        print(self.PERSON_LIST[np.argmax(y_pred)])
        # print("(0, 1): " + "(" + y_pred[0] + ", " + y_pred[1] + ") , " + self.PERSON_LIST[np.argmax(y_pred)])
        return self.PERSON_LIST[np.argmax(y_pred)]



    # Test this class using saved model and weights from classifier_images_faces.py
    def main(self):
        dg = DataGenerator
        self.validation_generator = dg.test_data_generator()
        x, y = dg.generate_data(self.validation_generator, 20)

        y_pred = self.model.predict(self.load(x[0]), batch_size=self.batch_size, verbose=1)
        # y_pred = self.model.predict(x, batch_size=self.batch_size, verbose=1)
        sys.exit(0)
        if(True):
          for i in range(len(y_pred)):
            print(x[i].shape)
            print(self.model)
            # sys.exit(0)

            y_correct = np.argmax(y[i])
            print(y[i])
            print(y_pred[i])
            y_pred_scalar = np.argmax(y_pred[i])
            print(y_pred)
            print("Correct answer: " + self.PERSON_LIST[y_correct])
            print("Prediction: " + self.PERSON_LIST[y_pred_scalar])
            print("###############")

            plt.imshow(x[i]/255)
            plt.show()