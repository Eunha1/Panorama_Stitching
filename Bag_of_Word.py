from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, classification_report
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import sklearn
import time
import warnings
warnings.filterwarnings("ignore")


def load_image (img_path):
    img = cv2.imread(img_path)
    return img
def statistic():
    label = []
    num_images = []

    for i in os.listdir('trainingset', ):
        label.append(i)
        num_images.append(len(os.listdir(os.path.join('trainingset',i))))
    return label, num_images
def read_data(label2id):
    X = []
    Y = []

    for label in os.listdir('trainingset'):
        for img_file in os.listdir(os.path.join('trainingset', label)):
            img = load_image(os.path.join('trainingset',label,img_file))
            X.append(img)
            Y.append(label2id[label])
    return X, Y
def extract_sift_features(X):
    image_descriptors = []
    sift = cv2.SIFT_create()

    for i in range(len(X)):
        kp, des = sift.detectAndCompute(X[i],None)
        image_descriptors.append(des)
    return image_descriptors
def all_descriptors (image_descriptors):
    all_descriptors = []
    for descriptors in image_descriptors:
         if descriptors is not None:
            for des in descriptors:
             all_descriptors.append(des)
    return all_descriptors
def kmeans_bow(all_descriptors, num_clusters):
    start = time.time()

    bow_dict = []
    kMeans = KMeans(num_clusters)
    kMeans.fit(all_descriptors)
    bow_dict = kMeans.cluster_centers_

    print('Process time: ', time.time() - start)

    return bow_dict
def create_features_bow(image_descriptors, BoW, num_clusters):
    X_features = []
    for i in range(len(image_descriptors)):

        features = np.array([0] * num_clusters)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)
            argmin = np.argmin(distance,axis=1)
            for j in argmin:
                features[j] += 1

        X_features.append(features)
    return X_features

samples_list = []
samples_label = []
for label in os.listdir('trainingset', ):
    sample_file = os.listdir(os.path.join('trainingset', label))[0]
    samples_list.append(load_image(os.path.join('trainingset', label, sample_file)))
    samples_label.append(label)
for i in range(len(samples_list)):
    plt.subplot(2, 3, i + 1), plt.imshow(cv2.cvtColor(samples_list[i], cv2.COLOR_BGR2RGB))
    plt.title(samples_label[i]), plt.xticks([]), plt.yticks([])
plt.show()

label, num_images = statistic()
y_pos = np.arange(len(label))
plt.barh(y_pos, num_images, align='center', alpha=0.5)
plt.yticks(y_pos, label)
plt.show()

label2id = {'pedestrian':0, 'moto':1, 'truck':2, 'car':3, 'bus':4}

X, Y = read_data(label2id)
image_descriptors = extract_sift_features(X)
all_descriptors = all_descriptors(image_descriptors)

num_clusters = 50
if not os.path.isfile('bow_dictionary4.pkl'):
    BoW = kmeans_bow(all_descriptors, num_clusters)
    pickle.dump(BoW, open('bow_dictionary.pkl', 'wb'))
else:
    BoW = pickle.load(open('bow_dictionary.pkl', 'rb'))

X_features = create_features_bow(image_descriptors, BoW, num_clusters)

X_train = []
X_test = []
Y_train = []
Y_test = []

X_train, X_test, Y_train, Y_test = train_test_split(X_features,Y,test_size=0.2,random_state=42)


svm = sklearn.svm.SVC(C = 30)
svm.fit(X_train, Y_train)
svm.score(X_test, Y_test)

Y_predict = svm.predict(X_test)
plot_confusion_matrix(svm, X_test, Y_test)
print(classification_report(Y_test, Y_predict, digits=3, target_names=list(label2id.keys())))


img = load_image('image_test/car.png')
my_X = [img]

my_image_descriptors = extract_sift_features(my_X)
my_X_features = create_features_bow(my_image_descriptors,BoW,50)

print(len(my_image_descriptors[0]))
print(my_X_features[0].shape)

y_pred = svm.predict(my_X_features)

print(y_pred)
print(label2id)
# Get your label name using label2id variable (define above)
for key, value in label2id.items():
    if value == y_pred[0]:
        print('Your prediction: ', key)