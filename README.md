Ce script Python template_main.py réalise le clustering de documents textuels en utilisant différentes techniques de réduction de dimensionnalité (ACP, UMAP, t-SNE) combinées à l'algorithme K-means pour regrouper les données. Nous avons également utilisé le spherical Kmeans et implémenté une fonction de cross-validation 

Fonctionnalités et Chargement des données : 
Les données sont chargées à partir du dataset 20 Newsgroups via scikit-learn. 
Embedding de Textes : Les embeddings des documents sont générés à l'aide du modèle SentenceTransformer. 
Réduction de Dimensionnalité : Les embeddings textuels sont réduits à 2 dimensions en utilisant l'une des techniques : ACP, UMAP, t-SNE. 
 Clustering : L'algorithme K-means et Spherical Kmeans sont appliqués sur les données réduites pour former des clusters.
 Évaluation : Les résultats de clustering sont évalués à l'aide des scores NMI (Normalized Mutual Information) et ARI (Adjusted Rand Index).

Utilisation :

Installation des Dépendances : Il faut installer les dépendances en utilisant pip install -r requirements.txt. Exécution : Exécutez le script Python pour charger les données, générer les embeddings, réduire la dimensionnalité, effectuer le clustering et évaluer les résultats.

python main.py Résultats : Les résultats de clustering sont imprimés dans la console, et des visualisations en 2D des clusters sont affichées pour chaque méthode de réduction de dimensionnalité.# examen_data_eng