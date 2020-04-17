'''Computes tf-idf and returns a matrix of documents and features
'''
import doc_loader
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdf:

    def __init__(self):
        self.documents = doc_loader.restore_documents()
        self.vectorizer = TfidfVectorizer()

    def get_matrix(self):
        contents = []
        for id, content in self.documents.items():
            doc_content = ' '.join(content)
            contents.append(doc_content)
        return self.vectorizer.fit_transform(contents)

    def get_vectorizer(self):
        return self.vectorizer


if __name__ == '__main__':
    tfidf = TfIdf()
    matrix = tfidf.get_matrix()
    print('compute tfidf')