'''Clusters documents with different algorithms
'''
from tfidf import TfIdf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Used an elbow plot to determine number of clusters
CLUSTER_NUMBER = 17


class Clustering:

    def __init__(self):
        tfidf = TfIdf()
        self.ids, self.matrix = tfidf.get_matrix()
        self.vectorizer = tfidf.get_vectorizer()

    def elbow(self):
        ''' Method used to select the right number of clusters for K-means
        :return: prints a plot
        '''
        score = []
        max = range(5, 20, 1)

        for cluster_size in max:
            print('Running cluster %s' % cluster_size)
            kmeans = KMeans(n_clusters=cluster_size, init='k-means++', n_init=1, verbose=5)
            inertia = kmeans.fit(self.matrix).inertia_
            score.append(inertia)

        f, ax = plt.subplots(1, 1)
        ax.plot(max, score, marker='o')
        ax.set_xlabel('Cluster centers')
        ax.set_xticks(max)
        ax.set_xticklabels(max)
        ax.set_ylabel('score')
        ax.set_title('Sores by clusters')
        plt.show()


    def kmeans(self):
        feature_names = self.vectorizer.get_feature_names()
        km = KMeans(n_clusters=CLUSTER_NUMBER, init='k-means++', max_iter=300, n_init=1, verbose=5)
        cluster_labels = km.fit_predict(self.matrix)
        sorted_centroids = km.cluster_centers_.argsort()[:, ::-1]

        for i in range(CLUSTER_NUMBER):
            print('Top words by cluster {}'.format(i))
            for index in sorted_centroids[i, :5]:
                print(' %s' % feature_names[index], end='')
            print()


if __name__ == '__main__':
    clustering = Clustering()
    clustering.kmeans()




