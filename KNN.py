import os
from sklearn.neighbors import KNeighborsClassifier
from torchvision import transforms
from torchvision.datasets import MNIST


def knn(neighbours):
    dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    dataset_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    data = dataset.data.numpy().reshape((60000, 784))
    targets = dataset.targets.numpy()

    test_data = dataset_test.data.numpy().reshape((10000, 784))
    test_targets = dataset_test.targets.numpy()

    knn_model = KNeighborsClassifier(n_neighbors=neighbours, metric='euclidean')
    knn_model.fit(data, targets)

    # score = knn_model.score(test_data, test_targets)
    # print("k=%d, accuracy=%.2f%%" % (neighbours, score * 100))

    test_predictions = knn_model.predict(test_data)
    correct = 0
    for i in range(10000):
        if test_predictions[i] == test_targets[i]:
            correct += 1
    print(f'Accuracy is: {correct/100}% with', neighbours, 'Nearest Neighbour(s).')


knn(1)
knn(3)
