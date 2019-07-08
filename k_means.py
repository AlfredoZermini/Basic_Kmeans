import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy


def distance(p, c):
    return (np.sqrt((p[0, :] - c[0]) ** 2 + (p[1, :] - c[1]) ** 2))


def minimize_error(p, n_class, previous_tot_err):
    # reset errors metric
    tot_err = 0

    # loop over clusters
    for k in range(n_class):
        dist[:, k] = distance(p, C_upd[:, k])

    # assign each point to the closest cluster
    cluster_idx = np.argmin(dist, axis=1)

    for k in range(n_class):
        # update centroids
        C_upd[:, k] = np.mean(p[:, cluster_idx == k], axis=1)
        err = distance(p[:, cluster_idx == k], C_upd[:, k])

        # add sum of error for each point to total error metric
        tot_err += np.sum(err)

    # evaluate the difference between total metric error of current and
    delta_tot_err = np.abs(tot_err - previous_tot_err)

    # save tot_err for the next loop
    previous_tot_err = tot_err

    return delta_tot_err, previous_tot_err, C_upd, cluster_idx


def plot(p, centroid, n_class, cl_idx, clusters_list):

    list_color = ['r', 'dodgerblue', 'g', 'grey', 'y']
    fig, ax = plt.subplots()

    for idx in range(n_class):

        # plot clusters
        ax.scatter(p[0, cl_idx == idx], p[1, cl_idx == idx],
                   s=1, c=list_color[idx])

        # add centroids
        ax.scatter(centroid[0,idx],
                   centroid[1,idx],
                   label=clusters_list[idx],
                   marker='X',
                   s=150,
                   c=list_color[idx],
                   edgecolor='k')
        ax.legend()

    # save plot
    ax.figure.savefig(os.path.join(os.getcwd(), 'plot'))


if __name__ == "__main__":

    # clusters names
    clusters = ['Adam', 'Bob', 'Charley', 'David', 'Edward']

    # Number of clusters
    K = len(clusters)

    # Coordinates of the centroids
    C = np.zeros([2, K])  # original centroids
    C[0, :] = np.array([-0.357, -0.055, 2.674, 1.044, -1.495])  # x axis
    C[1, :] = np.array([-0.253, 4.392, -0.001, -1.251, -0.090])  # y axis

    # define updated centroids
    C_upd = deepcopy(C)

    # load points from .csv file
    points = np.loadtxt('input.csv', delimiter=',', unpack=True)

    # number of points
    N = points.shape[1]

    # distances matrix
    dist = np.zeros((N, K))

    # cluster flags for each point
    cluster_index = np.zeros(N)

    # initialize error
    previous_tot_err = 0

    # write to file
    with open("OUTPUT.TXT", "w") as text_file:

        # loop until total metric reaches 0
        while True:
            delta_tot_err, previous_tot_err, C_upd, cluster_index = minimize_error(
                points, K, previous_tot_err)
            text_file.write('error = %.3f\n' % delta_tot_err)

            if(delta_tot_err == 0):
                break

        for index, name in enumerate(clusters):
            text_file.write('%s   %.3f   %.3f\n' %
                            (name, C_upd[0, index], C_upd[1, index],))

    # plot
    plot(points, C_upd, K, cluster_index, clusters)
