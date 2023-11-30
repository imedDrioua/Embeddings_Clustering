from flask import Flask, render_template, request
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import prince
import pandas as pd 
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Fonctions pour effectuer la réduction de dimension et le clustering
def dim_red(mat, p, method):
    if method == 'ACP':
        mat = pd.DataFrame(mat)
        acp = prince.PCA(n_components=p)
        red_mat = acp.fit_transform(mat).values
    elif method == 'TSNE':
        p = 3
        tsne = TSNE(n_components=p, random_state=42)
        red_mat = tsne.fit_transform(mat)
    elif method == 'UMAP':
        red_mat = UMAP(n_components=p).fit_transform(mat)
    else:
        raise Exception("Veuillez choisir l'une des trois méthodes : ACP, TSNE, UMAP")
    return red_mat

def visualize(data, labels, method):
    x = data[:, 0]
    y = data[:, 1]
    plt.figure(figsize=(15, 10))
    plt.scatter(x, y, c=labels)
    plt.title(f"Scatter Plot des Deux premières dimensions avec la méthode {method}")
    plt.xlabel('Première Colonne')
    plt.ylabel('Deuxième Colonne')
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

def clust(mat, k):
    kmeans = KMeans(n_clusters=k, n_init=1)
    pred = kmeans.fit_predict(mat)
    return pred

def clust_spherical_kmeans(mat, k):
    spherical_kmeans = SpectralClustering(n_clusters=3, affinity='nearest_neighbors')
    pred = spherical_kmeans.fit_predict(mat)
    return pred

def cross_validation(mat, k, num_iterations,labels):
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
    return avg_nmi, avg_ari, std_nmi, std_ari


# Route principale
@app.route('/')
def index():
    return render_template('index.html')

# Route pour le traitement des données
@app.route('/process_data', methods=['POST'])
def process_data():
    method = request.form['method']

    # Importation des données
    embeddings = pd.read_csv("dataset.csv")
    labels = embeddings.pop("label")
    k = len(set(labels))

    # Réduction de dimension et clustering
    red_emb = dim_red(embeddings, 2, method)
    pred = clust(red_emb, k)
    pred_sk = clust_spherical_kmeans(red_emb, k)

    # Évaluation des résultats de clustering
    nmi_score = normalized_mutual_info_score(pred, labels)
    nmi_score_sk = normalized_mutual_info_score(pred_sk, labels)
    ari_score = adjusted_rand_score(pred, labels)
    ari_score_sk = adjusted_rand_score(pred_sk, labels)

    # Cross-validation pour KMeans
    avg_nmi, avg_ari, std_nmi, std_ari = cross_validation(red_emb, k, 100,labels)

    # Visualisation et sauvegarde du plot
    plot_url = visualize(red_emb, pred, method)

    return render_template('result.html', method=method, nmi_score=nmi_score, ari_score=ari_score,
                           nmi_score_sk=nmi_score_sk, ari_score_sk=ari_score_sk, avg_nmi=avg_nmi,
                           avg_ari=avg_ari, std_nmi=std_nmi, std_ari=std_ari, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
