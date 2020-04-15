'''Clusters documents with different algorithms
'''
from tfidf import TfIdf
from sklearn.cluster import KMeans


class Clustering:
    def __init__(self):
        tfidf = TfIdf()
        self.matrix = tfidf.get_matrix()
        self.vectorizer = tfidf.get_vectorizer()

    def kmeans(self):
        feature_names = self.vectorizer.get_feature_names()
        km = KMeans(n_clusters=100, init='k-means++', max_iter=300, n_init=1, verbose=5)
        km.fit(self.matrix)
        ## print cluster centroids labels


if __name__ == '__main__':
    clustering = Clustering()
    clustering.kmeans()




