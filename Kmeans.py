import os
import numpy as np
from sklearn import metrics
from torchvision import transforms
from torchvision.datasets import MNIST
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans


def infer_cluster_labels(kMeans, actual_labels):
    inferred_labels = {}

    for i in range(kMeans.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(kMeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

    return inferred_labels


def infer_data_labels(X_labels, cluster_labels):
    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key

    return predicted_labels


dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
dataset_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

features = dataset.data.numpy().reshape((60000, 784))
true_labels = dataset.targets.numpy()

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 27,
}

kmeans = KMeans(n_clusters=10, **kmeans_kwargs)
kmeans.fit(features)

test_data = dataset_test.data.numpy().reshape((10000, 784))
test_targets = dataset_test.targets.numpy()

# determine cluster labels
cl_labels = infer_cluster_labels(kmeans, true_labels)
test_clusters = kmeans.predict(test_data)
pred_labels = infer_data_labels(test_clusters, cl_labels)

accuracy = metrics.accuracy_score(test_targets, pred_labels)
print(f'Accuracy is: {accuracy * 100 :.2f}% with K-Means.')
