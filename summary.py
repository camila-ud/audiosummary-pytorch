#-*- coding: utf-8 -*-
# @Author: Camila
#Audiosummary using soundnet and spectralclustering
import argparse

#--mesure time
import time 
# soundnet in pytorch
import soundnet.extract_features as ex
#load with soundfile
import soundfile as sf
import pandas as pd
# graphs
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from tsnecuda import TSNE

plt.style.use('bmh')
import os

from sklearn.metrics import silhouette_score,calinski_harabasz_score
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_distances
import joblib

def spectral_explore_cluster(features_vector,best_neighbors,max_cluster=10,version = False):
    metrics = []
    all_labels = []
    #Find number cluster    
    number = range(2,max_cluster +1)
    for i in tqdm(number): 
        #evaluate using calinski 
        spectral = SpectralClustering(n_clusters=i, 
                                      eigen_solver='arpack',
                                      affinity="nearest_neighbors",
                                       assign_labels="discretize",
                                      random_state=0,n_neighbors=best_neighbors,n_jobs = -1).fit((features_vector))

        labels = spectral.labels_
        metrics.append(calinski_harabasz_score(features_vector, labels))
        all_labels.append(labels)
        # calinski: the higher the better
    df = pd.DataFrame(metrics,index = number, columns = ['calinski'])
   # df.plot(y=['calinski'],subplots=True,sharex=True,figsize=(10,15),fontsize=16,linewidth=4)
   
    ##best_cluster: The index where the metrics is maximum.
    best_cluster = int(df.idxmax())
    print("Best number of cluster is {:d}".format(best_cluster))
    #check dispersion of labels
    if version:
        best_cluster = reduce_number_cluster_v2(all_labels[best_cluster - 2])
    else:
        best_cluster = reduce_number_cluster(all_labels[best_cluster - 2])
    
    return best_cluster

def reduce_number_cluster(labels):
    counter = []
    best = len(np.unique(labels))
    for i in np.unique(labels):
        #count by label 
        counter.append((len(labels[labels==i]))/len(labels))
    
    df = pd.Series(counter,index=np.unique(labels))
    if len(df[df < 0.1]) > 0:
        print(df.mean()/df.std())
        print(df)
        if (df.mean()/df.std()) < 1.25:
            #Highly dispersed data, we keep the labels 
            #that have more dots than the first quartile.
            best = len(df[df > df.quantile(0.25)])
            print("Highly dispersed, final number of clusters {}".format(best))    
    return best

def reduce_number_cluster_v2(labels):
    counter = []
    best = len(np.unique(labels))
    for i in np.unique(labels):
        #count by label 
        counter.append((len(labels[labels==i]))/len(labels))
    
    df = pd.Series(counter,index=np.unique(labels))
    best = len(df[df > 0.1])
    print("xxxxxxx {}".format(best))    
    return best

def spectral_explore_neighbors(features_vector,max_neighbors=30):
    metrics = []
    #Find number neighbors    
    if max_neighbors > features_vector.shape[0]:
        max_neighbors = features_vector.shape[0]//2 
    
    number = range(2,max_neighbors +1)
    for i in tqdm(number): 
        #fix n-cluster = 2, evaluate using calinski and silhouette
        spectral = SpectralClustering(n_clusters=2, 
                                      eigen_solver='arpack',
                                      affinity="nearest_neighbors",
                                      assign_labels="discretize",
                                      random_state=0,n_neighbors=i,n_jobs = -1).fit((features_vector))
        labels = spectral.labels_
        metrics.append([silhouette_score(features_vector, labels,metric='cosine'),  # silhouette : best 1, worst -1 
                        calinski_harabasz_score(features_vector, labels)])   # calinski: the higher the better
       
    df = pd.DataFrame(metrics,index = number, columns = ['silhouette','calinski'])
    df.plot(y=['silhouette','calinski'],subplots=True,sharex=True,figsize=(10,12),fontsize=14,linewidth=2)
        
    ##best_neighbors: The mean of the indices where the first derivative of the metrics is maximum.
    best_neighbors = int(round((df.diff().idxmax().mean())))
    print("Best number of neighbors is {:d}".format(best_neighbors))
    return best_neighbors

#to_utils
def get_values():
    to_time = {}
    with open('./soundnet/relation_layer_seconds.txt', 'r') as reader:
        for i in reader:
            key,m,b = i.split()
            if key != 'name_layer':
                to_time[key] = [float(m),float(b)]
    return to_time

def plot_two_axis(data_audio,data_clustering,name_layer,resultdir,duration=None):      
    fig, ax1 = plt.subplots(figsize=(14,7)) 
    t = (np.array(range(0,len(data_audio))))/22050  #vector_time to audio
    xlim = duration*22050 if (duration != None) else duration #seg --> to --> sample
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('audio')
    ax1.plot(t[:xlim], data_audio[:xlim])
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # 
    m,b = get_values()[name_layer] #get slope (m) and interception (b) for a given layer
    t = (np.array(range(0,len(data_clustering))))/m
    xlim = int((duration*m)+b) if (duration != None) else duration  #seg --> to --> label    
    ax2.set_ylabel('cluster', color = 'r')  # we already handled the x-label with ax1
    ax2.plot(t[:xlim], data_clustering[:xlim],'r')
    ax2.tick_params(axis='y',color='r')
    plt.title(name_layer+'non_b')
    plt.savefig(os.path.join(resultdir,'cluster.png'))
    plt.close()

def write_clustered_segments(data_audio,data_clustering,name_layer,resultpath):
   
    cursr = 22050
    m,b = get_values()[name_layer] #get slope (m) and interception (b) for a given layer
    seglength = int((1/m)*cursr)
    for i in range(data_clustering.max()+1): # Loop through all cluster labels
         # fetch all labels with the current value
        ind_curlabel = np.argwhere(data_clustering==i)
        # Calculate the offsets in the original wave file 
        offsets_wave = ind_curlabel * seglength        
        bigwave = []        
        
        for curoffset in tqdm(range(len(offsets_wave))):
            # cut the original wave file and save the excerpt
            curwave = data_audio[offsets_wave[curoffset][0]:(offsets_wave[curoffset][0]+seglength)]      
            if len(curwave) == 0:
                continue           
            bigwave.append(curwave)      
        bigwave = np.hstack(bigwave)
        sf.write(os.path.join(resultpath,'{}_summary_cluster_{}.wav'.format(name_layer,i)), bigwave, cursr)
    
def write(data_audio,data_clustering,name_layer,outputpath):
    resultdir = os.path.join(outputpath)
    os.makedirs(resultdir,exist_ok=True)
    write_clustered_segments(data_audio,data_clustering,name_layer,outputpath)

def extract(filepath,idlayer = 6):
    audio,sr = ex.load_audio(filepath)
    features = ex.extract_pytorch_feature(audio,'./soundnet/sound8.pth')   
    print("Features: \n")
    print([x.shape for x in features])    
    ##extract vector
    conv = ex.extract_vector(features,idlayer) #features vector   
    return audio,conv 
def make_clustering(conv,k,c):
    spectral = SpectralClustering(n_clusters=c, 
                                      eigen_solver='arpack',
                                      affinity="nearest_neighbors",
                                      assign_labels="discretize",
                                      random_state=0,n_neighbors=k).fit((conv))
    return spectral.labels_
def to_tsne(conv,labels,resultpath):
    plt.figure(figsize=(8,8))
    X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10,random_seed=0).fit_transform(conv)
    plt.scatter(X_embedded[:,0],X_embedded[:,1],c=labels,alpha=0.8)
    plt.colorbar()
    plt.title('TSNE Features and labels')
    plt.savefig(os.path.join(resultpath,'tsne.png'))
    plt.close()


def automatic_summary(inputfile,resultpath,plot=False):
    audiopath = inputfile
    #extract conv7
    name_layer = 'conv7'
    idlayer = 6
    
    ## extract features and find k and c
    audio,conv = extract(audiopath,idlayer) #conv7
    k = spectral_explore_neighbors(conv)
    c = spectral_explore_cluster(conv,k)
    ##make clustering
    labels = make_clustering(conv,k,c)

    #get slope (m) and interception (b) for a given layer
    m,b = get_values()[name_layer] 
    t = (np.array(range(0,len(labels))))/m    
    cluster = np.stack([t,labels]).T  #time and labels

    print(cluster.shape)
    #write segments 
    write(audio,labels,'conv7',resultpath)

    to_tsne(conv,labels,resultpath)  #tsne
    print("tsne exported")
    if plot:
        plot_two_axis(audio,labels,name_layer,resultpath) #plot
        print("plot exported")
    np.savez_compressed("{}/features_conv{}.npz".format(resultpath,idlayer+1),
                        feature_vector = conv,idlayer = idlayer,labels = cluster)
    
    del audio,conv,labels,cluster

def automatic_summary_v2(inputfile,resultpath,
                        plot = True,name_layer = 'conv7',
                        idlayer = 6):
    audiopath = inputfile
    #extract conv7
    
    
    ## extract features and find k and c
    audio,conv = extract(audiopath,idlayer) #conv7
    k = spectral_explore_neighbors(conv)
    c = spectral_explore_cluster(conv,k,version = True)
    ##make clustering
    labels = make_clustering(conv,k,c)

    #get slope (m) and interception (b) for a given layer
    m,b = get_values()[name_layer] 
    t = (np.array(range(0,len(labels))))/m    
    cluster = np.stack([t,labels]).T  #time and labels

    print(cluster.shape)
    #write segments 
    write(audio,labels,name_layer,resultpath)

    to_tsne(conv,labels,resultpath)  #tsne
    print("tsne exported")
    if plot:
        plot_two_axis(audio,labels,name_layer,resultpath) #plot
        print("plot exported")
    np.savez_compressed("{}/features_conv_v2{}.npz".format(resultpath,idlayer+1),
                       feature_vector = conv,idlayer = idlayer,labels = cluster)
    
    del audio,conv,labels,cluster

                    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resultpath', metavar = "resultpath", help='Folder where to store the results', default="results")
    parser.add_argument('--inputfile', metavar = 'inputfile', help='Input file name to be processed (should be audio file)')
    parser.add_argument('--save', metavar = 'save', help='Bool to save plot')
    args, _ = parser.parse_known_args()
    return args
    


def main(args):
    automatic_summary(args.inputfile,args.resultpath,args.save)
    
if __name__=="__main__":
    args = get_parser()
    main(args)
