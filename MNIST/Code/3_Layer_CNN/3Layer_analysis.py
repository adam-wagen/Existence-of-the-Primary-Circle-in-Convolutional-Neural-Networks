# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:07:29 2021

@author: Adam
"""
import os
import numpy as np
import matplotlib 
matplotlib.use('agg') 
import matplotlib.pyplot as plt
print("imported pyplot");
import persim
from persim import plot_diagrams
#import pytz
import mlxtend
#os.chdir('/Analysis')
#os.chdir('C:\\Users\\Adam\\Documents\\College\\Grad_School\\Research\\MNIST\\Dissertation_Analysis\\New_3l_Analysis_full\Analysis')
#os.chdir('3_Layer_Analysis/Analysis')
#runList = [20,40,60,100,150,200];
#runList = [60];
runList = [1,2,3,4,5,10,15,20,40,60,100,200];
cubesList = [30];
#overlapList = np.arange(0.1,1,0.1);
overlapList = [0.66];
overlap = 0.66
kVal = 100;
rho = 0.1;
for runs in runList:
    for cubes in cubesList:
        for overlap in overlapList:
            #runs = 40;
            #cubes = 30;
            #overlap = 0.66;
            #cd "C:\Users\Adam\Documents\College\Grad_School\Research\MNIST"
            data = np.genfromtxt("CNNweights_"+ str(runs) +".csv",delimiter=',')
            #data = np.genfromtxt("CNNweights_"+ str(runs) +"_l3.csv",delimiter=',')
            #data = np.genfromtxt("C:/Users/Adam/Documents/College/Grad_School/Research/MNIST/Dissertation_Analysis/New_3l_Analysis_full/CNNweights_"+ str(runs) +".csv",delimiter=',')
            #data = np.genfromtxt("C:/Users/Adam/Documents/College/Grad_School/Research/Cats_Dogs/Analysis/3_Layer_Analysis/CNNweights_"+ str(runs) +".csv",delimiter=',')
            import sklearn
            from sklearn import preprocessing
            
            from mlxtend.preprocessing import MeanCenterer
            #data = np.genfromtxt('CNNweights-new.csv',delimiter=',')
            #data = np.genfromtxt('/content/drive/Shareddrives/Clingher-Wagenknecht/Work/Python/CNN-MNIST/Adam-Python/CNNweights.csv',delimiter=',')
            #data = np.genfromtxt('GCweights.csv',delimiter=',')
            meanData = MeanCenterer().fit_transform(data)
            scaler = preprocessing.StandardScaler().fit(data)
            normalizer = preprocessing.Normalizer().fit(data)
            normData = normalizer.transform(meanData)
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
              #indexes = order(mat[,k], decreasing=FALSE)
              #indexes = np.argsort(mat[:,k])
              indexes = np.argsort(dists)
              print("got the indxs")
              # we return the the rho* number of points in X closest points
              return(X[indexes[0:math.ceil(nrows*rho)],:])
            
            
            
            # K-Nearest Neighbor Density Filter the points using K=200 and taking the closest 30%
            pts = knndf(normData,200,0.3)
            
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca_pts = pca.fit_transform(pts)
            
            from gtda.plotting import plot_point_cloud
            pcaplt = plot_point_cloud(pca_pts)
            #,filename="PCA"+str(runs) + "_cubes" + str(cubes) + "_overlap" + str(overlap) +".png")
            pcaplt.write_image("PCA_"+str(runs) + "_cubes" + str(cubes) + "_overlap" + str(overlap) +".png");
            
            from ripser import Rips

            #rips = Rips()
            #diagrams = rips.fit_transform(pca_pts)
            #ripsplt = rips.plot(diagrams)
            #plot_diagrams(diagrams, show=True)

            rips = Rips()
            diagrams = rips.fit_transform(pca_pts)
            plot_diagrams(diagrams, show=True, fileName="Persistence_"+str(runs) + "_k" + str(kVal) + "_rho" + str(rho) +".png")
            plt.clf();
    

            from sklearn.cluster import DBSCAN
            import kmapper as km
            # Initialize
            mapper = km.KeplerMapper(verbose=2)
            
            # Fit to and transform the data
            
            projected_data = mapper.fit_transform(pts, projection=PCA(n_components=2),scaler=None) # X-Y axis
            cover = km.Cover(n_cubes=cubes, perc_overlap=overlap)
            #cluster = DBSCAN(min_samples=7)
            cluster = sklearn.cluster.AgglomerativeClustering(n_clusters = None,compute_full_tree = True, distance_threshold = 15.0, linkage='single')
            # Define the simplicial complex
            graph = mapper.map(projected_data,pts,clusterer=cluster,cover=cover)
            
            
            
            # Visualize it
            mapper.visualize(graph,
                             path_html="./keplermapper_output_"+str(runs) + "_cubes" + str(cubes) + "_overlap" + str(overlap) + ".html", 
                             title="MNIST " + str(runs) + " Epochs" + " cubes " + str(cubes) + " overlap " + str(overlap))
            
            #import networkx as nx
            #nx_graph = km.adapter.to_nx(graph)
            #nx.draw(nx_graph)
            from kmapper.plotlyviz import plotlyviz
            plotlyviz(graph, 
                      graph_layout='fr',
                      title="MNIST " + str(runs) + " Epochs" + " cubes " + str(cubes) + " overlap " + str(overlap), 
                      filename="Plot_"+str(runs) + "_cubes" + str(cubes) + "_overlap" + str(overlap) +".png")
            plt.clf();


runList = [1,2,3,4,5,10,15,20,40,60,100];
#runList = [40,60,80,100];

runList = [20];
cubesList = [30];
overlapList =[0.66];
for runs in runList:
    for cubes in cubesList:
        for overlap in overlapList:
            #runs = 40;
            #cubes = 30;
            #overlap = 0.66;
            #cd "C:\Users\Adam\Documents\College\Grad_School\Research\MNIST"
            
            #data = np.genfromtxt("C:/Users/Adam/Documents/College/Grad_School/Research/MNIST/CNNweights_"+ str(runs) +".csv",delimiter=',')
            #data = np.genfromtxt("C:/Users/Adam/Documents/College/Grad_School/Research/Cats_Dogs/Analysis/3_Layer_Analysis/CNNweights_"+ str(runs) +"_l2.csv",delimiter=',')
            #data = np.genfromtxt("C:/Users/Adam/Documents/College/Grad_School/Research/MNIST/Dissertation_Analysis/New_3l_Analysis_full/CNNweights_"+ str(runs) +"_l2.csv",delimiter=',')
            data = np.genfromtxt("CNNweights_"+ str(runs) +"_l2.csv",delimiter=',')
            #data = data[0:round(len(data)/3)]
            #data = np.genfromtxt('CNNweights-new.csv',delimiter=',')
            #data = np.genfromtxt('/content/drive/Shareddrives/Clingher-Wagenknecht/Work/Python/CNN-MNIST/Adam-Python/CNNweights.csv',delimiter=',')
            #data = np.genfromtxt('GCweights.csv',delimiter=',')
            meanData = MeanCenterer().fit_transform(data)
            scaler = preprocessing.StandardScaler().fit(data)
            normalizer = preprocessing.Normalizer().fit(data)
            normData = normalizer.transform(meanData)
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
              #indexes = order(mat[,k], decreasing=FALSE)
              #indexes = np.argsort(mat[:,k])
              indexes = np.argsort(dists)
              print("got the indxs")
              # we return the the rho* number of points in X closest points
              return(X[indexes[0:math.ceil(nrows*rho)],:])
            
            
            
            # K-Nearest Neighbor Density Filter the points using K=200 and taking the closest 30%
            pts = knndf(normData,200,0.1)
            
            
            pca = PCA(n_components=2)
            pca_pts = pca.fit_transform(pts)
            
            
            pcaplt = plot_point_cloud(pca_pts)
            #,filename="PCA"+str(runs) + "_cubes" + str(cubes) + "_overlap" + str(overlap) +".png")
            pcaplt.write_image("PCA_l2"+str(runs) + "_cubes" + str(cubes) + "_overlap" + str(overlap) +".png");
            
            

            #rips = Rips()
            #diagrams = rips.fit_transform(pca_pts)
            #ripsplt = rips.plot(diagrams)
            #plot_diagrams(diagrams, show=True)
            
            
            rips = Rips()
            diagrams = rips.fit_transform(pca_pts)
            plot_diagrams(diagrams, show=True, fileName="Persistence_l2_"+str(runs) + "_k" + str(kVal) + "_rho" + str(rho) +".png")
            plt.clf();

    
            
            # Initialize
            mapper = km.KeplerMapper(verbose=2)
            
            # Fit to and transform the data
            
            projected_data = mapper.fit_transform(pts, projection=PCA(n_components=2),scaler=None) # X-Y axis
            cover = km.Cover(n_cubes=cubes, perc_overlap=overlap)
            #cluster = DBSCAN(min_samples=7)
            cluster = sklearn.cluster.AgglomerativeClustering(n_clusters = None,compute_full_tree = True, distance_threshold = 15.0, linkage='single')
            # Define the simplicial complex
            graph = mapper.map(projected_data,pts,clusterer=cluster,cover=cover)
            
            
            colors = np.ones(1920)*(1/1920)
            # Visualize it
            mapper.visualize(graph,
                              path_html="./keplermapper_output_l2_"+str(runs) + "_cubes" + str(cubes) + "_overlap" + str(overlap) + ".html", 
                              title="MNIST " + str(runs) + " Epochs" + " cubes " + str(cubes) + " overlap " + str(overlap))
            
            #import networkx as nx
            #nx_graph = km.adapter.to_nx(graph)
            #nx.draw(nx_graph)
            from kmapper.plotlyviz import plotlyviz
            plotlyviz(graph, 
                      graph_layout='fr',
                      title="MNIST " + str(runs) + " Epochs" + " cubes " + str(cubes) + " overlap " + str(overlap), 
                      filename="Plot_l2"+str(runs) + "_cubes" + str(cubes) + "_overlap" + str(overlap) +".png")
            plt.clf();

#runList = [40,60];
for runs in runList:
    for cubes in cubesList:
        for overlap in overlapList:
            #runs = 40;
            #cubes = 30;
            #overlap = 0.66;
            #cd "C:\Users\Adam\Documents\College\Grad_School\Research\MNIST"
            
            #data = np.genfromtxt("C:/Users/Adam/Documents/College/Grad_School/Research/MNIST/CNNweights_"+ str(runs) +".csv",delimiter=',')
            #data = np.genfromtxt("C:/Users/Adam/Documents/College/Grad_School/Research/Cats_Dogs/Analysis/3_Layer_Analysis/CNNweights_"+ str(runs) +"_l3.csv",delimiter=',')
            #data = np.genfromtxt("C:/Users/Adam/Documents/College/Grad_School/Research/MNIST/Dissertation_Analysis/New_3l_Analysis_full/CNNweights_"+ str(runs) +"_l3.csv",delimiter=',')
            data = np.genfromtxt("CNNweights_"+ str(runs) +"_l3.csv",delimiter=',')
            #data = data[0:round(len(data)/3)]
            #data = np.genfromtxt('CNNweights-new.csv',delimiter=',')
            #data = np.genfromtxt('/content/drive/Shareddrives/Clingher-Wagenknecht/Work/Python/CNN-MNIST/Adam-Python/CNNweights.csv',delimiter=',')
            #data = np.genfromtxt('GCweights.csv',delimiter=',')
            meanData = MeanCenterer().fit_transform(data)
            scaler = preprocessing.StandardScaler().fit(data)
            normalizer = preprocessing.Normalizer().fit(data)
            normData = normalizer.transform(meanData)
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
              #indexes = order(mat[,k], decreasing=FALSE)
              #indexes = np.argsort(mat[:,k])
              indexes = np.argsort(dists)
              print("got the indxs")
              # we return the the rho* number of points in X closest points
              return(X[indexes[0:math.ceil(nrows*rho)],:])
            
            
            
            # K-Nearest Neighbor Density Filter the points using K=200 and taking the closest 30%
            pts = knndf(normData,200,0.1)
            
            
            pca = PCA(n_components=2)
            pca_pts = pca.fit_transform(pts)
            
            
            pcaplt = plot_point_cloud(pca_pts)
            #,filename="PCA"+str(runs) + "_cubes" + str(cubes) + "_overlap" + str(overlap) +".png")
            pcaplt.write_image("PCA_l3"+str(runs) + "_cubes" + str(cubes) + "_overlap" + str(overlap) +".png");
            
            

            #rips = Rips()
            #diagrams = rips.fit_transform(pca_pts)
            #ripsplt = rips.plot(diagrams)
            #plot_diagrams(diagrams, show=True)
            
            
            rips = Rips()
            diagrams = rips.fit_transform(pca_pts)
            plot_diagrams(diagrams, show=True, fileName="Persistence_l3_"+str(runs) + "_k" + str(kVal) + "_rho" + str(rho) +".png")
            plt.clf();

    
            
            # Initialize
            mapper = km.KeplerMapper(verbose=2)
            
            # Fit to and transform the data
            
            projected_data = mapper.fit_transform(pts, projection=PCA(n_components=2),scaler=None) # X-Y axis
            cover = km.Cover(n_cubes=cubes, perc_overlap=overlap)
            #cluster = DBSCAN(min_samples=7)
            cluster = sklearn.cluster.AgglomerativeClustering(n_clusters = None,compute_full_tree = True, distance_threshold = 15.0, linkage='single')
            # Define the simplicial complex
            graph = mapper.map(projected_data,pts,clusterer=cluster,cover=cover)
            
            
            colors = np.ones(1920)*(1/1920)
            # Visualize it
            mapper.visualize(graph,
                              path_html="./keplermapper_output_l3_"+str(runs) + "_cubes" + str(cubes) + "_overlap" + str(overlap) + ".html", 
                              title="MNIST " + str(runs) + " Epochs" + " cubes " + str(cubes) + " overlap " + str(overlap))
            
            #import networkx as nx
            #nx_graph = km.adapter.to_nx(graph)
            #nx.draw(nx_graph)
            from kmapper.plotlyviz import plotlyviz
            plotlyviz(graph, 
                      graph_layout='fr',
                      title="MNIST " + str(runs) + " Epochs" + " cubes " + str(cubes) + " overlap " + str(overlap), 
                      filename="Plot_l3"+str(runs) + "_cubes" + str(cubes) + "_overlap" + str(overlap) +".png")
            plt.clf();


