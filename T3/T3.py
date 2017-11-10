# -*- coding: utf-8 -*-
"""
asdasdasdas
"""

import itertools
import logging
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA


# Activates Verbose on all models.
DEBUG = 0

# Name of the dataset directory.
DATASET_PATH = 'documents/data.csv'

# CSV FILE HAS 19924 ROWS AND 2209 COLUMNS
# EACH ROW REPRESENTS A DOCUMENT


def predict_kmedoids_labels(clusters, n):
    labels = labels = np.array([0 for x in range(n)])
    for i, rows in clusters.items():
        for j in rows:
            labels[j] = i
    return labels


def kMedoids(data, k, tmax=100):
    # determine dimensions of distance matrix data
    m, n = data.shape

    if k > n:
        raise Exception('too many medoids')

    # randomly initialize an array of k medoid indices
    medoids = np.arange(n)
    np.random.shuffle(medoids)
    medoids = np.sort(medoids[:k])

    # create a copy of the array of medoid indices
    new_medoids = np.copy(medoids)

    # initialize a dictionary to represent clusters
    clusters = {}


    for t in range(tmax):
        i = 0
        taken_values = np.zeros(n)
        for kappa in medoids:
            clusters[i] = np.array([kappa])
            taken_values[kappa] = 1
            i += 1

        # determine clusters, i. e. arrays of data indices
        J = np.argmin(data[:,medoids], axis=1)
        for kappa in range(k):
            neighbors = np.where(J==kappa)[0]
            if len(neighbors) > 1:
                clusters[kappa] = neighbors
                for i in clusters[kappa]:
                    if taken_values[i] == 1 and i != medoids[kappa]:
                        index = np.argwhere(clusters[kappa]==i)
                        clusters[kappa] = np.delete(clusters[kappa], index)
                    taken_values[i] = 1
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(data[np.ix_(clusters[kappa],clusters[kappa])],axis=1)
            j = np.argmin(J)
            new_medoids[kappa] = clusters[kappa][j]
        np.sort(new_medoids)
        # check for convergence
        if np.array_equal(medoids, new_medoids):
            break
        medoids = np.copy(new_medoids)
    else:
        # final update of cluster memberships
        J = np.argmin(data[:,medoids], axis=1)
        for kappa in range(k):
            clusters[kappa] = np.where(J==kappa)[0]

    # return results
    return medoids, clusters

def load_model_data(num_rows = 0, num_features = 0, ignore_features=[]):
    features = np.loadtxt(open(DATASET_PATH, 'rb'), delimiter=',', skiprows=1)
    if (num_rows != 0):
        features = features[range(num_rows)]

    n_deleted = 0
    ignore_features.sort()
    for i in ignore_features:
        features = np.delete(features, i - n_deleted, axis=1)
        n_deleted += 1

    if (num_features != 0):
        features = features[:, range(num_features)]


    return features


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

###
# Plots the comparison between the predicted and the actual value.
###
def plot_model_values(values):
    fig, ax = plt.subplots()
    y = list(range(len(values)))
    cmap = get_cmap(len(values[0]))
    for i in range(len(values[0])):
        ax.scatter(values[:,i], y, c = cmap(i))
    ax.legend(range(len(values[0])))
    ax.grid(True)
    plt.show()


###
# Plots the loss curve of the MLP model.
###
def plot_mpl_loss_curve(mlp):
    fig, ax = plt.subplots()
    ax.set_title("Neural Network Loss Rate")
    ax.plot(mlp.loss_curve_, c='red', linestyle='-')
    plt.show()


###
# This function prints and plots the confusion matrix.
###
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    np.set_printoptions(precision=2)
    plt.figure()

    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()


###
# Calculates the accuracy of a model.
###
def accuracy(y_test, y_pred):
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            count += 1
    acc = 100 * count / len(y_pred)
    return acc


###
# Main function, executes the model prediction.
###
def main():
    features = load_model_data(num_features=100, num_rows=10000, ignore_features=[0, 3, 4, 5, 8])
    #plot_model_values(features)

    # Reduces the data to 2 components only.
    #features = PCA(n_components=2).fit_transform(features)

    # distance matrix
    distanceVector = pairwise_distances(features, metric='euclidean')

    clusters_range = range(2,3)
    silhouette_scores = np.empty(len(clusters_range))
    silhouette_labels = {}
    silhouette_medoids = {}
    for n_clusters in clusters_range:
        # Execute the clustering with k-medoids.
        medoids, clusters = kMedoids(distanceVector, n_clusters)
        labels = predict_kmedoids_labels(clusters, len(features))

        # Uncomment to use k-means instead.
        #clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        #labels = clusterer.fit_predict(features)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(distanceVector, labels)
        silhouette_scores[n_clusters - clusters_range[0]] = silhouette_avg
        silhouette_labels[n_clusters] = labels
        silhouette_medoids[n_clusters] = medoids
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

    # Plots the average silhouette growth.
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 7)
    ax.plot(clusters_range, silhouette_scores)
    ax.set_title('Average silhouette plot for varying k')
    ax.set_xlabel("The number of clusters (k)")
    ax.set_ylabel("The average silhouette coefficient values")
    # The horizontal line for the best score 0.
    ax.axhline(0, color="red", linestyle="--")
    plt.show()

    # Calculating the best scores to extract metrics and analysis.
    best_index = np.argmax(silhouette_scores)
    best_n_clusters = best_index + clusters_range[0]
    best_score = silhouette_scores[best_index]
    best_labels = silhouette_labels[best_n_clusters]
    best_medoids = silhouette_medoids[best_n_clusters]


    # Print the medoids and their 4 closest neighbors
    # J = np.argmin(distanceVector[:,best_medoids], axis=1)
    # for k in range(len(best_medoids)):
    #     print "Medoid %d: %d" % (k, best_medoids[k])
    #     neighbors = clusters[k]
    #     print "Closest neighbors: %s" % neighbors[:5]
    #     print "-------------------------"

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.6, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(features) + (n_clusters + 1) * 10])

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(distanceVector, best_labels)

    y_lower = 10
    for i in range(best_n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[best_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / best_n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters. AVG: %.3f" % best_score)
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=best_score, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(best_labels.astype(float) / best_n_clusters)
    ax2.scatter(distanceVector[:, 0], distanceVector[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    centers = distanceVector[best_medoids]

    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data as distance vector.")
    ax2.set_xlabel("1st component of the distane vector")
    ax2.set_ylabel("2nd component of the distane vector")

    plt.suptitle(("Silhouette analysis for KMedoids clustering on sample data "
                  "with n_clusters = %d" % best_n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()

###
# Sets logging information and calls main function.
###
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main()
