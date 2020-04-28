import doc_finder as df
import doc_loader as dl
import constants

def main():
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


if __name__ == '__main__':
    main()