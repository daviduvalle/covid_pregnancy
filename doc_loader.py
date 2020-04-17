'''Preprocess documents and saves them in binary
format using pickle
'''
import json
import string
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

FILE_LIST = 'pregnancy_files.txt'
PREPROCESSED_DATA = 'document_content'


def load_docs():
    '''
    :return: a dictionary doc_id, body_text of documents
    '''
    file_list = open(FILE_LIST, 'r')
    documents = {}
    for file in file_list:
        file = file.replace('\n', '')
        with open(file, 'r') as json_file:
            raw_document = json.load(json_file)
            content = raw_document['body_text']
            doc_id = raw_document['paper_id']
            text = ''
            for text_line in content:
                text = text + text_line['text'] + ' '
            documents[doc_id] = text
    return documents


def preprocess_docs(docs):
    '''Common text preprocessing tasks
    :param docs: raw docs
    :return: preprocessed docs
    '''
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    for doc_id, content in docs.items():
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
        filtered = [word for word in word_tokens if word not in stop_words]
        filtered = [lemmatizer.lemmatize(word, pos='v') for word in filtered]

        docs[doc_id] = filtered

    return docs


def save_documents(docs):
    '''Saves data in binary format
    :param docs: pre-processed docs
    '''
    with open(PREPROCESSED_DATA, 'wb') as preprocessed_data:
        pickle.dump(docs, preprocessed_data)


def restore_documents():
    '''Restores saved documents into memory
    :return: a dictionary of docs and their data
    '''
    with open(PREPROCESSED_DATA, 'rb') as preprocessed_data:
        docs = pickle.load(preprocessed_data)
    return docs


def main():
    documents = load_docs()
    documents = preprocess_docs(documents)
    save_documents(documents)


if __name__ == '__main__':
    main()
