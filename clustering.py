'''Clusters documents with different algorithms
'''
from tfidf import TfIdf
from sklearn.cluster import KMeans

# Used Silhouette analysis to pick the number of clusters
CLUSTER_NUMBER = 10


class Clustering:

    def __init__(self):
        tfidf = TfIdf()
        self.matrix = tfidf.get_matrix()
        self.vectorizer = tfidf.get_vectorizer()

    def kmeans(self):
        feature_names = self.vectorizer.get_feature_names()
        km = KMeans(n_clusters=CLUSTER_NUMBER, init='k-means++', max_iter=300, n_init=1, verbose=5)
        cluster_labels = km.fit_predict(self.matrix)
        sorted_centroids = km.cluster_centers_.argsort()[:, ::-1]

        for i in range(CLUSTER_NUMBER):
            print('Cluster {}'.format(i))
            for index in sorted_centroids[i, :5]:
                print(' %s' % feature_names[index], end='')
            print()


if __name__ == '__main__':
    clustering = Clustering()
    clustering.kmeans()




