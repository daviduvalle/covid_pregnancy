import doc_finder as df
import doc_loader as dl
import constants
from kmeans import KMeans
import sys
import os


def main(args):
    run_kmeans_only = False
    if len(args) > 0 and args[0] == 'algorithm_only':
        print('Running algorithm only, assuming preprocessed files exist')
        if not os.path.exists(constants.OUTPUT_PATH):
            raise Exception('Output directory needs to exists, try running again without flags')
        if not os.path.exists(constants.PREPROCESSED_DOCS):
            raise Exception('Preprocessed docs pickly file needs to exists, try running again without flags')
        run_kmeans_only = True

    if not run_kmeans_only:
        print('Running K-means. The process takes a few minutes')
        files = df.filter_dataset()
        print('Found %d matching files' % len(files))
        print('Saving file into: %s' % constants.FOUND_DOCS)
        df.save_files(files)
        documents = dl.load_docs()
        print('Pre-processing documents before vectorization')
        documents = dl.preprocess_docs(documents)
        dl.save_documents(documents)
        print('Pre-processed documented saved into: %s' % constants.PREPROCESSED_DOCS)
        print('Running the actual algorithm...')

    kmeans = KMeans()
    centroids, labels, feature_names = kmeans.run()
    kmeans.plot(labels, constants.VISUALIZATION)
    cluster_keyword = kmeans.topn_words(centroids, feature_names, 5)
    cluster_doc = kmeans.cluster_to_doc(labels)
    kmeans.print_top_words(cluster_keyword)
    kmeans.print_docs_by_cluster(cluster_doc)
    print('Writing report at: %s' % constants.FINAL_REPORT)
    kmeans.save_file(cluster_doc, cluster_keyword)


if __name__ == '__main__':
    main(sys.argv[1:])