'''Preprocess documents and saves them in binary
format using pickle
'''
import json
import string
import re
import pickle
import constants
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

PREPROCESSED_DATA = 'document_content.pickle'


def load_docs():
    '''
    :return: a dictionary doc_id, body_text of documents
    '''
    file_list = open(constants.FOUND_DOCS, 'r')
    documents = {}
    for file in file_list:
        file = file.replace('\n', '')
        with open(file, 'r') as json_file:
            raw_document = json.load(json_file)
            content = raw_document['body_text']
            doc_id = raw_document['paper_id']
            title = raw_document['metadata']['title']
            text = ''
            for text_line in content:
                text = text + text_line['text'] + ' '
            documents[doc_id] = (title, text)
    return documents


def preprocess_docs(docs):
    '''Common text preprocessing tasks
    :param docs: raw docs
    :return: preprocessed docs
    '''
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    for doc_id, title_content in docs.items():
        title, content = title_content
        # lower case
        content = content.lower()
        # remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        content = content.translate(translator)
        # remove numbers
        content = re.sub(r'\d+', '', content)
        # remove extra white spaces
        content = ' '.join(content.split())
        word_tokens = word_tokenize(content)
        cleaned_content = [word for word in word_tokens if word not in stop_words]
        cleaned_content = [lemmatizer.lemmatize(word, pos='v') for word in cleaned_content]

        docs[doc_id] = (title, cleaned_content)

    return docs


def save_documents(docs):
    '''Saves data in binary format
    :param docs: pre-processed docs
    '''
    with open(constants.PREPROCESSED_DOCS, 'wb') as preprocessed_data:
        pickle.dump(docs, preprocessed_data)


def restore_documents():
    '''Restores saved documents into memory
    :return: a dictionary of docs and their data
    '''
    with open(constants.PREPROCESSED_DOCS, 'rb') as preprocessed_data:
        docs = pickle.load(preprocessed_data)
    return docs


def main():
    documents = load_docs()
    documents = preprocess_docs(documents)
    save_documents(documents)


if __name__ == '__main__':
    main()
