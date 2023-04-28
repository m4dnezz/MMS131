import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


############################################
# Made by Niclas Persson 2023/04/26
###########################################


def importdata(name: str):
    alldata = np.genfromtxt(name)
    xdata = alldata[:, 0]
    ydata = alldata[:, 1]
    return xdata, ydata, alldata


def minowski_distance(x1, x2, i=1):
    if i == 1:
        return sum(abs(val1 - val2) for val1, val2 in zip(x1, x2)) # Manhattan

    elif i == 2:
        return np.sqrt(np.abs((x1[0] - x2[0])) ** 2 + np.abs((x1[1] - x2[1]))**2) # Euclidean

    elif i == 3:
        return np.cbrt(np.sum(np.abs((x1 - x2)) ** 3))

    else:
        raise ValueError("i must be 1 or 2")


def kmeans(data, k, centroids, max_iters=10000):
    labels = None
    C1 = []
    C2 = []
    C3 = []
    # Iterate until convergence or max_iters is reached
    for step in range(max_iters):

        # Assign samples to nearest centroid
        distances = np.array([[minowski_distance(data, c, i=1) for c in centroids] for data in data])

        # np.argmin function returns the index of the nearest centroid to each sample, resulting in the labels array
        labels = np.argmin(distances, axis=1)

        # Update centroids
        # new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        new_centroids = np.array([np.median(data[labels == i], axis=0) for i in range(k)])

        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < 0.001:
            break

        C1.append(centroids[0])
        C2.append(centroids[1])
        C3.append(centroids[2])
        centroids = new_centroids

    C1.append(centroids[0])
    C2.append(centroids[1])
    C3.append(centroids[2])
    C = [C1, C2, C3]
    return centroids, labels, C


def knn(data, labels, point: tuple, k: int, i: int):
    k_closest_points = []
    point_labels = []
    # Calculate distance between new point and existing points
    distances = np.array([minowski_distance(point, c, i=i) for c in data])

    # Get the index of the k the closest points
    for m in range(k):
        cp = np.argmin(distances)
        k_closest_points.append(cp)
        # can't remove index since next index would be wrong. Assign inf instead
        distances[cp] = np.inf

    for points in k_closest_points:
        point_labels.append(labels[points] + 1)

    point_label = max(set(point_labels), key=point_labels.count)

    return point_label


def write_to_file(labels):
    file = "kmeans_labels.txt"
    with open(file, 'w') as file:
        for label in labels:
            file.write(f"{label + 1}\n")


def plotting(xdata, ydata, sc, test_point, centroidsx, centroidsy, C):
    fig, ax = plt.subplots()
    ax.scatter(test_point[0], test_point[1], label="test-point", color="black")
    ax.scatter(xdata, ydata, label="Data")
    ax.scatter(sc[0][0], sc[0][1], label="C1 - Start", color="green")
    ax.scatter(sc[1][0], sc[1][1], label="C2 - Start", color="red")
    ax.scatter(sc[2][0], sc[2][1], label="C3 - Start", color="yellow")
    ax.scatter(centroidsx[0], centroidsy[0], label="C1 - Final", color="green")
    ax.scatter(centroidsx[1], centroidsy[1], label="C2 - Final", color="red")
    ax.scatter(centroidsx[2], centroidsy[2], label="C3 - Final", color="yellow")
    fig.legend(loc="upper right")
    line_segments = LineCollection(C, color="black", linestyles="dotted")
    ax.add_collection(line_segments)
    plt.title("Plot of data and clustering")
    plt.show()


def main(file: str, sc, k=3):
    test_point = (0, 0)
    xdata, ydata, alldata = importdata(file)
    final_centroids, labels, C = kmeans(alldata, k, centroids=sc)
    write_to_file(labels)
    centroidsx, centroidsy = final_centroids.T
    plotting(xdata, ydata, sc, test_point, centroidsx, centroidsy, C)

    # print(knn(alldata, labels, test_point, k=3, i=1))
    # print(knn(alldata, labels, test_point, k=7, i=1))
    # print(knn(alldata, labels, test_point, k=11, i=1))
    #
    # print(knn(alldata, labels, test_point, k=3, i=2))
    # print(knn(alldata, labels, test_point, k=7, i=2))
    # print(knn(alldata, labels, test_point, k=11, i=2))
    #
    # print(knn(alldata, labels, test_point, k=3, i=3))
    # print(knn(alldata, labels, test_point, k=7, i=3))
    # print(knn(alldata, labels, test_point, k=11, i=3))




if __name__ == "__main__":
    filename = "data_kmeans.txt"
    starting_centroids = [[0, 0], [0, 1], [0, 2]]
    main(filename, starting_centroids)
