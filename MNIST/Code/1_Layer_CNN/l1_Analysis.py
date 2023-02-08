# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:07:29 2021

@author: Adam
"""
import os
import numpy as np
from persim import plot_diagrams
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing            
from mlxtend.preprocessing import MeanCenterer
from sklearn.decomposition import PCA
from gtda.plotting import plot_point_cloud
from ripser import Rips
from sklearn.cluster import DBSCAN
import kmapper as km
from kmapper.plotlyviz import plotlyviz

runList = [1,2,3,4,5,10,15,20,40,60,100,200];

cubesList = [30];
overlapList = [0.66];
overlap = 0.66
kVal = 100;
rho = 0.3;
for runs in runList:
    for cubes in cubesList:
        for overlap in overlapList:
            
            data = np.genfromtxt("CNNweights_"+ str(runs) +".csv",delimiter=',')
            
            normData = preprocessing.scale(data,axis=1)
            
            # Define a function to perform the density filtration
            def knndf(X,k,rho):
              # compute the number of 9 dimensional points in the data set X
              nrows = len(X);
              
              import numpy as np
              import math
            
              from sklearn import neighbors
              tree = neighbors.KDTree(X)
            
              dists = [];
              
              
            
              for idx in range(len(X)):
                dist,ind = tree.query(X[idx:idx+1],k=k)
                dists = np.concatenate((dists,dist[:,k-1]))
            
            
              # The k-th column of each row is now the distance to the 200th closest point to the point i (i.e. k-th nearest neighbor)
              # we know select the indices of this matrix in increasing order of the k-th distance
              indexes = np.argsort(dists)
              
              # we return the the rho* number of points in X closest points
              return(X[indexes[0:math.ceil(nrows*rho)],:])
            
            
            
            # K-Nearest Neighbor Density Filter the points using K=200 and taking the closest 30%
            pts = knndf(normData,kVal,rho)
            
            pca = PCA(n_components=2)
            pca_pts = pca.fit_transform(pts)
            
            pcaplt = plot_point_cloud(pca_pts)
            
            pcaplt.write_image("PCA"+str(runs) + "_cubes" + str(cubes) + "_overlap" + str(overlap) +".png");
            
            rips = Rips()
            diagrams = rips.fit_transform(pca_pts)
            plot_diagrams(diagrams, show=True, fileName="Persistence_"+str(runs) + "_k" + str(kVal) + "_rho" + str(rho) +".png")
            plt.clf();
                

            # Initialize
            mapper = km.KeplerMapper(verbose=2)
            
            # Fit to and transform the data
            
            projected_data = mapper.fit_transform(pts, projection=PCA(n_components=2),scaler=None) # X-Y axis
            cover = km.Cover(n_cubes=cubes, perc_overlap=overlap)
            
            cluster = sklearn.cluster.AgglomerativeClustering(n_clusters = None,compute_full_tree = True, distance_threshold = 15.0, linkage='single')
            # Define the simplicial complex
            graph = mapper.map(projected_data,pts,clusterer=cluster,cover=cover)
            
            
            
            # Visualize it
            mapper.visualize(graph,
                             path_html="./keplermapper_output_"+str(runs) + "_cubes" + str(cubes) + "_overlap" + str(overlap) + ".html", 
                             title="MNIST " + str(runs) + " Epochs" + " cubes " + str(cubes) + " overlap " + str(overlap))
            
            #import networkx as nx
            
            plotlyviz(graph, 
                      graph_layout='fr',
                      title="MNIST " + str(runs) + " Epochs" + " cubes " + str(cubes) + " overlap " + str(overlap), 
                      filename="Plot"+str(runs) + "_cubes" + str(cubes) + "_overlap" + str(overlap) +".png")
            plt.clf();

