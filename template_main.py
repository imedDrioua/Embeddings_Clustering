from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np
import umap
from sklearn.cluster import KMeans
import prince
import pandas as pd 
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
        
    elif method=='AFC':
        red_mat = mat[:,:p]
        
    elif method=='UMAP':
        red_mat = umap.UMAP(n_components=p).fit_transform(mat)
        
    else:
        raise Exception("Please select one of the three methods : APC, AFC, UMAP")
    
    return red_mat

def visualize(data,labels): 
    x = data[:,0]
    y = data[:,1]
    plt.figure(figsize=(15,10))
    plt.scatter(x, y, c=pred)  # Vous pouvez changer 'viridis' à d'autres cartes de couleur (colormaps)
    plt.title('Scatter Plot des Deux deux première dimensions ')
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

# import data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# Perform dimensionality reduction and clustering for each method
methods = ['ACP', 'AFC', 'UMAP']
for method in methods:
    # Perform dimensionality reduction
    red_emb = dim_red(embeddings, 2, method)

    # Perform clustering
    pred = clust(red_emb, k)

    # Evaluate clustering results
    nmi_score = normalized_mutual_info_score(pred, labels)
    ari_score = adjusted_rand_score(pred, labels)

    # Print results
    print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')
    visualize(red_emb,pred)
        
