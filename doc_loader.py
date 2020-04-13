import json
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

FILE_LIST_PATH = 'pregnancy_files.txt'


def load_docs():
    '''
    :return: a dictionary doc_id, body_text of documents
    '''
    file_list = open(FILE_LIST_PATH, 'r')
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
    stop_words = set(stopwords.words('english'))
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
        docs[doc_id] = filtered

    return docs


def main():
    documents = load_docs()
    documents = preprocess_docs(documents)


if __name__ == '__main__':
    main()