'''Clusters documents with different algorithms
'''
from tfidf import TfIdf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio
import plotly.express as px

# Used the elbow method to determine number of clusters
CLUSTER_NUMBER = 17


class Clustering:

    def __init__(self):
        pio.renderers.default = 'browser'
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
        ''' Runs k-means and prints the top keywords per cluster
        :param num_keywords: number of top keywords to print
        :return centroids, labels, tf-idf features
        '''
        feature_names = self.vectorizer.get_feature_names()
        km = KMeans(n_clusters=CLUSTER_NUMBER, init='k-means++', n_init=1, verbose=5)
        cluster_labels = km.fit_predict(self.matrix)
        sorted_centroids = km.cluster_centers_.argsort()[:, ::-1]

        return sorted_centroids, cluster_labels, feature_names

    def plot(self, labels, html_path):
        dense = self.matrix.todense()
        pca = PCA(n_components=3).fit_transform(dense)
        norm = plt.Normalize(np.min(labels), np.max(labels))
        normalized = norm(labels)
        fig = px.scatter_3d(x=pca[:,0], y=pca[:,1], z=pca[:,2], color=normalized)
        fig.write_html(html_path)
        print('Clustering visualization using k-means saved as: {}'.format(html_path))

    def save_file(self, centroids, labels, feature_names, file_path):
        print('Top 10 words by cluster')
        cluster_to_doc = {}
        for i in range(CLUSTER_NUMBER):
            print('cluster #:{}'.format(i))
            cluster_keywords = ''
            for index in centroids[i, :10]:
                cluster_keywords += feature_names[index] + ' '
            print(cluster_keywords)
            print('labels {}'.format(labels.size))
            cluster_to_doc[i] = list()
            for e in range(0, labels.size):
                if labels[e] == i:
                    cluster_to_doc[i].append(self.ids[e])

        print('Docs by cluster')
        for i in range(CLUSTER_NUMBER):
            print('Cluster {} docs {}'.format(i, len(cluster_to_doc[i])))


if __name__ == '__main__':
    clustering = Clustering()
    centroids, labels, feature_names = clustering.kmeans()
    clustering.plot(labels, 'output.html')
    clustering.save_file(centroids, labels, feature_names, 'clusters.json')

