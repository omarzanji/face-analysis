import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow import keras
import tensorflow as tf

from PIL import Image

MODEL = '1000_100epoch.model'

class FaceLandmark:

    def __init__(self, test=0):
        if test: return
        self.data_dir = 'data/dataset_1000/'
        files = os.listdir(self.data_dir)
        self.landmarks = []
        self.segs = []
        self.images = []
        for f in files:
            f = str(f)
            if 'ldmks.txt' in f:
                landmarks = self.data_dir+f
                ldmks = np.loadtxt(landmarks)
                self.landmarks.append(ldmks.flatten())
            elif '_seg.png' in f:
                seg = self.data_dir+f
                seg_img = Image.open(seg) 
                seg_img = seg_img.getdata()
                seg_img = np.array(seg_img).reshape((512, 512))
                self.segs.append(seg_img)
            elif '.png' in f and not '_seg' in f:
                image = self.data_dir+f
                img = Image.open(image) 
                img = img.getdata()
                img = np.array(img).reshape((512, 512, 3)) 
                self.images.append(img)

        self.x = self.images
        self.y = self.landmarks


    def plot_sample_landmarks(self, sample_num):
        if len(str(sample_num))==1:
            sample = '00000'+str(sample_num)
        else: sample = '0000'+str(sample_num)
        dir = 'data/dataset_100/'
        sample_list = os.listdir(dir)
        for s in sample_list:
            s = str(s)
            if sample+'_ldmks.txt' in s:
                landmarks = dir+s
            elif sample+'_seg.png' in s:
                seg = dir+s
            elif sample+'.png' in s:
                image = dir+s
        img = mpimg.imread(image)
        ldmks = np.loadtxt(landmarks)
        plt.imshow(img)
        for xy in ldmks:
            x = xy[0]
            # print(x)
            y = xy[1]
            # print(y)
            plt.plot(x, y, '.', color='red', markersize=3) 
        self.img = img
        plt.show()


    def create_train_model(self):
        xtrain, xtest, ytrain, self.ytest = train_test_split(np.array(self.x), np.array(self.y))
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(6, (7,7), padding='same', input_shape=(512,512,3)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

            tf.keras.layers.Conv2D(16, (3,3), padding='same'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

            tf.keras.layers.Conv2D(120, (7,7), padding='same'),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(140),
            tf.keras.layers.Activation('relu')
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics='accuracy')
        self.hist = self.model.fit(xtrain, ytrain, epochs=300)
        self.ypreds = self.model.predict(xtest)
        accuracy = self.model.evaluate(xtest, self.ytest)
        print(f'\n\naccuracy: {accuracy[1]}\n\n')


    def load_predict(self):
        image = 'omar.png'
        image_raw = Image.open(image) 
        image_raw = image_raw.getdata()
        image_raw = np.array(image_raw).reshape((1, 512, 512, 3))
        self.image_raw = image_raw
        self.model = keras.models.load_model(MODEL)
        self.ypred = self.model.predict(image_raw)
        
        print(self.ypred)
        np.save('omar_pred.npy', self.ypred)

    def plot_pred(self):
        image = 'omar.png'
        plt_img = mpimg.imread(image)
        plt.imshow(plt_img)
        self.ldmks = np.load('omar_pred.npy').reshape((70,2))
        for xy in self.ldmks:
            x = xy[0]
            print(x)
            y = xy[1]
            print(y)
            plt.plot(x, y, '.', color='green', markersize=5) 
        
        plt.show()


if __name__ == "__main__":
    
    face = FaceLandmark(test=0)
    # face.load_predict()

    # face.plot_pred()

    face.create_train_model()

    # 55, 12, and 99 are pretty good.
    # face.plot_sample_landmarks(12)
