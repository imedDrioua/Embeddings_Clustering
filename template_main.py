from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np
import umap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import prince
import pandas as pd 
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt




def dim_red(mat, p, method):
    '''
    Perform dimensionality reduction

    Input:
    -----
        mat : NxM list 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''
    if method=='ACP':
        mat=pd.DataFrame(mat)
        acp = prince.PCA(n_components=p)
        red_mat = acp.fit_transform(mat).values
        
    elif method=='TSNE':
        p=3
        tsne = TSNE(n_components=p, random_state=42)
        red_mat= tsne.fit_transform(mat)
        
    elif method=='UMAP':
        red_mat = umap.UMAP(n_components=p).fit_transform(mat)
        
    else:
        raise Exception("Please select one of the three methods : APC, AFC, UMAP")
    
    return red_mat

def visualize(data,labels,mehtod): 
    x = data[:,0]
    y = data[:,1]
    plt.figure(figsize=(15,10))
    plt.scatter(x, y, c=pred)  # Vous pouvez changer 'viridis' à d'autres cartes de couleur (colormaps)
    plt.title(f"Scatter Plot des Deux deux première dimensions avec la méthode {method}" )
    plt.xlabel('Première Colonne')
    plt.ylabel('Deuxième Colonne')
    
    # Affichage du plot
    plt.show()
        
def clust(mat, k):
    '''
    Perform clustering

    Input:
    -----
        mat : input list 
        k : number of cluster
    Output:
    ------
        pred : list of predicted labels
    '''

    
    kmeans = KMeans(n_clusters=k)
    pred = kmeans.fit_predict(mat)

    
    return pred
def cross_validation(mat, k, num_iterations):
    '''
    Execute clustering function multiple times with different initializations

    Input:
    -----
        mat : input list 
        k : number of clusters
        num_iterations : number of times to run the clustering function
    Output:
    ------
        avg_nmi : average normalized mutual info score
        avg_ari : average adjusted rand score
        std_nmi : standard deviation of NMI scores
        std_ari : standard deviation of ARI scores
    '''

    nmis = []
    aris = []

    for _ in range(num_iterations):
        pred = clust(mat, k)
        nmi_score = normalized_mutual_info_score(pred, labels)
        ari_score = adjusted_rand_score(pred, labels)

        nmis.append(nmi_score)
        aris.append(ari_score)

    avg_nmi = np.mean(nmis)
    avg_ari = np.mean(aris)
    std_nmi = np.std(nmis)
    std_ari = np.std(aris)
    print(f'NMI: {avg_nmi:.2f} \nARI: {avg_ari:.2f} \nSTD_NMI: {std_nmi} \n \nSTD_ARI: std_ari')
    

def clust_spherical_kmeans(mat, k):
    '''
    Perform clustering

    Input:
    -----
        mat : input list 
        k : number of cluster
    Output:
    ------
        pred : list of predicted labels
    '''
    spherical_kmeans = SpectralClustering(n_clusters=3, affinity='nearest_neighbors')
    pred= spherical_kmeans.fit_predict(mat)
    
    return pred

# import data
embeddings = pd.read_csv("dataset.csv")
labels = embeddings.pop("label")
k = len(set(labels))

# Perform dimensionality reduction and clustering for each method
methods = ['ACP', 'TSNE', 'UMAP']
for method in methods:
    # Perform dimensionality reduction
    red_emb = dim_red(embeddings, 2, method)

    # Perform clustering
    pred = clust(red_emb, k)
    pred_sk=clust_spherical_kmeans(red_emb,k)
    # Evaluate clustering results
    nmi_score = normalized_mutual_info_score(pred, labels)
    nmi_score_sk = normalized_mutual_info_score(pred_sk, labels)
    ari_score = adjusted_rand_score(pred, labels)
    ari_score_sk = adjusted_rand_score(pred_sk, labels)
    # Print results
    print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')
    print(f'Using Kmeans Clustering, Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')
    print(f'Using SphericalKmeans Clustering, Method: {method}\nNMI: {nmi_score_sk:.2f} \nARI: {ari_score_sk:.2f}\n')
    print(f'Using Kmeans Clustering, Method: {method}\n {cross_validation(red_emb, k, 100)}')
    visualize(red_emb,pred,method)
      

