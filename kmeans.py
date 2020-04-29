'''Clusters documents with different algorithms
'''
from tfidf import TfIdf
from sklearn.decomposition import PCA
import constants
import sklearn.cluster
import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio
import plotly.express as px

# Used the elbow method to determine number of clusters
CLUSTERS = 17


class ClusterDoc:
    def __init__(self, id, description, docs):
        self.id = id
        self.description = description
        self.docs = docs


class KMeans:

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
            kmeans = sklearn.cluster.KMeans(n_clusters=cluster_size, init='k-means++', n_init=1, verbose=5)
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

    def run(self):
        ''' Runs k-means and prints the top keywords per cluster
        :param num_keywords: number of top keywords to print
        :return centroids, labels, tf-idf features
        '''
        feature_names = self.vectorizer.get_feature_names()
        km = sklearn.cluster.KMeans(n_clusters=CLUSTERS, init='k-means++', random_state=5, n_init=1, verbose=5)
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

    def topn_words(self, centroids, feature_names, n):
        ''' Returns the top N keywords by cluster
        :param centroids: identified by kmeans
        :param feature_names: maps back to keywords
        :param n: number of words to describe a cluster
        :return: a dictionary of clusters and top N keywords
        '''
        cluster_topwords = {}
        for i in range(CLUSTERS):
            cluster_keywords = ''
            for index in centroids[i, :n]:
                cluster_keywords += feature_names[index] + ' '
            cluster_topwords[i] = cluster_keywords
        return cluster_topwords

    def print_top_words(self, cluster_topwords):
        for cluster_id, words in cluster_topwords.items():
            print('{}: {}'.format(cluster_id, words))

    def print_docs_by_cluster(self, cluster_doc):
        for cluster in sorted(cluster_doc, key=lambda k :len(cluster_doc[k]), reverse=True):
            print('{}, docs: {}'.format(cluster, len(cluster_doc[cluster])))

    def cluster_to_doc(self, labels):
        cluster_to_doc = {}
        for i in range(CLUSTERS):
            cluster_to_doc[i] = list()
            for e in range(0, labels.size):
                if labels[e] == i:
                    cluster_to_doc[i].append(self.ids[e])

        return cluster_to_doc

    def save_file(self, cluster_doc, cluster_keyword, file_path):
        output_list = []
        for cluster_id in sorted(cluster_doc, key=lambda k: len(cluster_doc[k]), reverse=True):
            doc = ClusterDoc(cluster_id, cluster_keyword[cluster_id], cluster_doc[cluster_id])
            output_list.append(doc)

        print('Final list to write {}'.format(len(output_list)))


if __name__ == '__main__':
    kmeans = KMeans()
    centroids, labels, feature_names = kmeans.run()
    kmeans.plot(labels, constants.VISUALIZATION)
    cluster_keyword = kmeans.topn_words(centroids, feature_names, 5)
    cluster_doc = kmeans.cluster_to_doc(labels)
    kmeans.print_top_words(cluster_keyword)
    kmeans.print_docs_by_cluster(cluster_doc)
    kmeans.save_file(cluster_doc, cluster_keyword, 'output.json')

