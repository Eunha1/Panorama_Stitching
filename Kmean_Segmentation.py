from sklearn.cluster import MeanShift, KMeans
from IPython.display import Image
from skimage.filters import threshold_multiotsu
import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def init_seed(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]


def kmean(img, n_clusters):
    X = img.reshape((-1, 3))
    random_seeds = init_seed(X, 30)
    X_float = np.float32(X)
    #Đưa ra tiêu chí để làm giảm độ phức tạp tính toán
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    compactness, labels, (centers) = cv2.kmeans(X_float, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    # Tính toán ảnh phân loại
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)
    return segmented_image,compactness


img = cv2.imread('images/img1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 13))
noRegion = [2,4,6,8,10]

plt.subplot(3,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.xticks([]), plt.yticks([])
count1 = 1
for k in noRegion:
    img1,compactness = kmean(img,k)
    plt.subplot(3,2,count1 + 1)
    plt.imshow(img1)
    plt.title("No of regions = {}".format(k))
    plt.xticks([]), plt.yticks([])
    count1 += 1
#plt.subplots_adjust()
plt.subplots_adjust(wspace=0, hspace=0.2)
plt.show()

noRegion = range(2, 16)

objective = []
for k in noRegion:
    img1,compactness = kmean(img,k)
    objective.append(compactness)
    plt.plot(objective)
plt.show()