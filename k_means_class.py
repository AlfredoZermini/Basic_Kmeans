import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy

class K_means(object):
    def __init__(self, filename):
        self.filename = filename

    # load data
    def load_data(self):
        return np.loadtxt(self.filename, delimiter=',', unpack=True)

    # define distance metric
    def distance(self, p ,c):
        return (np.sqrt((p[0,:]-c[0])**2+(p[1,:]-c[1])**2 ) )

    # reset errors metrics
    def initialize_params(self, te, pte):

        if te == 'reset': # reset metrics
            te = 0
            pte = 0
            dte = 1000
        else:
            dte = np.abs(te-pte) # evaluate the difference between total metric error of current and previous steps
            pte = te

        return te, dte, pte

    # minimum distance between points and centroid
    def find_clusters(self, d):
        return np.argmin(d, axis = 1)

    # points belonging to a given cluster
    def points_to_cluster(self, p, cluster, idx):
        return p[:,cluster == idx]

    # update centroids coordinates
    def update_centroids(self, cluster, idx):
        return np.mean(points[:,cluster == idx], axis=1)

    # plot points with different colors
    def display(self, p, c, n_class, cl):

        list_color = ['r', 'b', 'g', 'm', 'y']
        fig, ax = plt.subplots()

        for idx in range(n_class):
            # plot cluster
            ax.scatter( self.points_to_cluster(p, cl, idx)[0,:], self.points_to_cluster(p, cl, idx)[1,:], s=1, c=list_color[idx])

            # add centroids
            ax.scatter(c[0,idx], c[1,idx], marker = 'P', s=150, c=list_color[idx], edgecolor = 'k')

        # save plot
        ax.figure.savefig(os.path.join(os.getcwd(), 'plot'))

if __name__ == "__main__":
    # Number of clusters
    K = 5

    # Coordinates of the centroids
    C = np.zeros([2,K]) # original centroids
    C[0,:] = np.asarray([-0.357, -0.055, 2.674, 1.044, -1.495]) # x axis
    C[1,:] = np.asarray([-0.253, 4.392, -0.001, -1.251, -0.090]) # y axis

    # updated centroids
    C_upd = deepcopy(C)

    # load points
    kmeans = K_means('input.csv')
    points = kmeans.load_data()

    # define number of points
    N = points.shape[1]

    # define matrix which contains the distance between each point and each of the K centroids
    dist = np.zeros((N,K))

    # define vector which contains the cluster index for each point
    cluster_index = np.zeros(N)

    # initialize errors metrics
    tot_err, delta_tot_err, prev_tot_err = kmeans.initialize_params('reset',None)

    ### Loop until the total error metric value stops changing
    while delta_tot_err != 0:

        # reset metrics
        tot_err, delta_tot_err, __ = kmeans.initialize_params('reset',None)

        # loop over clusters
        for k in range(K):
           dist[:,k] = kmeans.distance(points,C_upd[:,k])

        # assign each point to the closest cluster
        cluster_index = kmeans.find_clusters(dist)

        for k in range(K):
            # update centroids
            C_upd[:,k] = kmeans.update_centroids(cluster_index,k)

            # calculate errors
            err = kmeans.distance(kmeans.points_to_cluster(points,cluster_index,k),C_upd[:,k])

            # add sum of error for each point to total error metric
            tot_err += np.sum(err)

        # save tot_err for the next loop
        __, delta_tot_err, prev_tot_err = kmeans.initialize_params(tot_err, prev_tot_err)
        print delta_tot_err

    ### PLOT POINTS
    kmeans.display(points, C_upd, K, cluster_index)
