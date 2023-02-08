# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 20:32:13 2022

@author: Adam
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


import matplotlib.cm as cm
from persim import plot_diagrams
import sklearn
from sklearn import preprocessing
            
from mlxtend.preprocessing import MeanCenterer
#import pytz

def PlotMeanGrid(X):
    numEls = len(X);
    meanGrid = [0,0,0,0,0,0,0,0,0];
    for idx in range(numEls):
        meanGrid += X[idx]
    
    #print(meanGrid)
    meanGrid = meanGrid/numEls;
    print(meanGrid)
    print(meanGrid.reshape((3,3)))
    plt.imshow(meanGrid.reshape((3,3)), origin='upper', cmap=cm.Greys_r);
    plt.show();

runs = 1;
cubes = 30;
overlap = 0.66;
kVal = 200;
rho = 0.3;

os.chdir('../Results/Weights/2_Layer_CNN')

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
pts = knndf(normData,200,0.3)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_pts = pca.fit_transform(pts)

from gtda.plotting import plot_point_cloud
pcaplt = plot_point_cloud(pca_pts)

from ripser import Rips

rips = Rips()
diagrams = rips.fit_transform(pca_pts)

from sklearn.cluster import DBSCAN
import kmapper as km
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
                 path_html="./keplermapper_cycle_analysis_"+str(runs) + "_cubes" + str(cubes) + "_overlap" + str(overlap) + ".html", 
                 title="MNIST " + str(runs) + " Epochs" + " cubes " + str(cubes) + " overlap " + str(overlap))

# Use the following commands to plot the mean of the grid points traced back from cluster nodes in the html mapper diagram
#nodeList = graph['nodes']
#sortedList = sorted(nodeList, key=lambda k: len(nodeList[k]), reverse=True)
#PlotMeanGrid(pts[nodeList['cube310_cluster0']])