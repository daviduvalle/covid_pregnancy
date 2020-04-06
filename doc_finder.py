''' Finds relevant documents and creates a file index
for other programs to locate documents
'''
import os

BASE_DIR = 'covid'
search_words = set()
search_words.add('pregnant')
search_words.add('pregnancy')
search_words.add('neonatal')


def filter_dataset():
    '''Loads the dataset
    :return: list of documents
    '''
    for dirpath, _, _ in os.walk(BASE_DIR):
        matching_documents = set()
        for file in os.listdir(dirpath):
            file_path = dirpath + '/' + file
            if os.path.isfile(file_path) and '.json' in file:
                content = open(file_path, 'r')
                content_string = content.read().lower()
                for search_word in search_words:
                    if search_word in content_string:
                        matching_documents.add(file_path)
                content.close()
    return matching_documents


def save_files(files):
    '''Saves the found files in a new CSV files used later for
    fast retrieval
    :param files: list of found files
    '''
    output = open('pregnancy_files.txt', 'w')
    for file in files:
        output.write(file+'\n')
    output.close()


def main():
    files = filter_dataset()
    save_files(files)

if __name__ == '__main__':
    main()