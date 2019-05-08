from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import os
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import math
import Image
import numpy as np
from matplotlib import pyplot as plt
import face_utilities
import random
from sklearn import metrics
from prettytable import PrettyTable

epsilon = 95

# prepare model
# https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(250,250, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.6))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.6))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

model.load_weights(os.path.join('resources','vgg_face_weights.h5'))

vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

def preprocess_image(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def get_eucl_distance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def verify(img1, img2, personNumber1, personNumber2, print_plot=False):
    img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))[0, :]
    img2_representation = vgg_face_descriptor.predict(preprocess_image(img2))[0, :]
    euclidean_distance = get_eucl_distance(img1_representation, img2_representation)
    if (euclidean_distance < epsilon):
        if (personNumber1 == personNumber2):
            label = "Rovnaka osoba"
            ret = 1
        else: # (personNumber1 != personNumber2):
            label = "Same person (wrong)\nDistance= {}".format(euclidean_distance)
            ret = 0
    else:
        if (personNumber1 == personNumber2):
            label = "Different person (wrong)\nDistance= {}".format(euclidean_distance)
            ret = 0
        else: # (personNumber1 != personNumber2):
            label = "Iná osoba"
            ret = 1
    if print_plot:
        f = plt.figure()

        f.add_subplot(1, 2, 1)
        plt.imshow(img1)
        plt.xlabel('Person nr.'+str(personNumber1))
        plt.title(label)

        f.add_subplot(1, 2, 2)
        plt.imshow(img2)
        plt.xlabel('Person nr.'+str(personNumber2))
        plt.show(block=True)
    return ret, round(euclidean_distance,2)

def evaluate_pairs(img1, img2, personNumber1, personNumber2, pairs):
    img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))[0, :]
    img2_representation = vgg_face_descriptor.predict(preprocess_image(img2))[0, :]
    euclidean_distance = get_eucl_distance(img1_representation, img2_representation)
    if (euclidean_distance < epsilon):
        if (personNumber1 == personNumber2):
            pairs.truePositive()
            ret = 1
        else:
            pairs.falsePositive()
            ret = 0
    else:
        if (personNumber1 == personNumber2):
            pairs.falseNegative()
            ret = 0
        else:
            pairs.trueNegative()
            ret = 1
    return ret, euclidean_distance

def get_variance(personNr):
    personFolderPath = os.path.join(face_utilities.validaFaceFolder, str(personNr))
    distances = []
    for index1, imageName1 in enumerate(os.listdir(personFolderPath)):
        for index2, imageName2 in enumerate(os.listdir(personFolderPath)):
            if index1 < index2:
                img1 =  face_utilities.transform_image(Image.open(os.path.join(personFolderPath, imageName1)), imageName1)
                img2 =  face_utilities.transform_image(Image.open(os.path.join(personFolderPath, imageName2)), imageName2)
                _, distance = verify(img1, img2, personNr, personNr, print_plot=False)
                distances.append(math.floor(distance))
    return np.mean(distances), np.var(distances)

def recalculate_face_success():
    correct_ans=0
    total_ans=0
    for personNumber in range(1,101):
        count = 0
        print("Processing person nr.{}".format(personNumber))
        for personNumber2 in range(personNumber,101):
            firstPersonImg = face_utilities.transform_random_image(personNumber)
            secondPersonImg = face_utilities.transform_random_image(personNumber2)
            correct_ans,_ = correct_ans + verify(firstPersonImg,secondPersonImg, personNumber, personNumber2, print_plot=False)[0]
            total_ans = total_ans + 1
            if total_ans%10==0:
                break
    print("Success {}%".format(math.floor(correct_ans / ((total_ans) / 100))))      #87%

def compare(firstPersonNr, secondPersonNr):
    firstPersonFace, secondPersonFace =face_utilities.get_two_transformed_faces(firstPersonNr, secondPersonNr)
    _,distance = verify(firstPersonFace,secondPersonFace, firstPersonNr, secondPersonNr, print_plot=True)
    meanVal, variance = get_variance(firstPersonNr)
    zScore = (math.floor(distance)-meanVal)/variance

    return round(zScore,2), distance, firstPersonFace, secondPersonFace

def convert_distance_to_percentage(distance):
    if distance==0:
        return 100
    elif distance<30:
        return 90
    elif distance<60:
        return 80
    elif distance<75:
        return 75
    elif distance<90:
        return 65
    elif distance<100:
        return 35
    elif distance<120:
        return 20
    elif distance<140:
        return 10
    else:
        return 5

def print_ROC():
    pairs = Pairs()
    labels = []
    distances = []
    for iteration in range(1, 100):
        personNumber1 = random.randint(1,100)
        personNumber2 = random.randint(1,100)
        firstPersonImg1 = face_utilities.transform_random_image(personNumber1)
        firstPersonImg2 = face_utilities.transform_random_image(personNumber1)
        secondPersonImg = face_utilities.transform_random_image(personNumber2)

        ret, euclidean_distance = evaluate_pairs(firstPersonImg1, firstPersonImg2, personNumber1, personNumber1, pairs)
        distances.append(euclidean_distance)
        labels.append(ret)
        ret, euclidean_distance = evaluate_pairs(firstPersonImg1, secondPersonImg, personNumber1, personNumber2, pairs)
        distances.append(euclidean_distance)
        labels.append(ret)

    pairs.printPairs()
    distances, labels  = zip(*sorted(zip(distances, labels )))

    y = np.array(labels)
    scores = np.array(distances)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(fpr, tpr,color='darkorange', lw=2 )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Face ROC')
    plt.show()
    print(pairs)

class Pairs:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
    def truePositive(self):
        self.tp = self.tp + 1
    def falsePositive(self):
        self.fp = self.fp + 1
    def falseNegative(self):
        self.fn = self.fn + 1
    def trueNegative(self):
        self.tn = self.tn + 1
    def printPairs(self):
        t = PrettyTable(['', 'True', 'False'])
        t.add_row(['True', 'TP: '+str(self.tp), 'FP: '+str(self.fp)])
        t.add_row(['False', 'FN: '+str(self.fn), 'TN: '+str(self.tn)])
        print(t)
