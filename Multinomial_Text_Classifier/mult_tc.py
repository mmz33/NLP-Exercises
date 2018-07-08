# python3.6 run.py 'dataset name'

import numpy as np
import pickle
import os.path, os
import matplotlib.pyplot as plt
import texttable as tt
import pandas as pd # Used only to plot confusion matrix
import sys
import argparse

test_sizes = [500, 1000, 5000, 10000, 20000, 93508]

class Document:
    """Represents a document

    Attributes:
        true_class: The true class of the document
        learned_class: The class learned from the training
        words_count: Dictionary from a word to its frequency
    """

    true_class = ''
    learned_class = ''
    words_count = {}

    def __init__(self, true_class, words_count):
        self.true_class = true_class
        self.words_count = words_count

    def get_true_class(self):
        return self.true_class

    def set_learned_class(self, l_class):
        self.learned_class = l_class

    def get_learned_class(self):
        return self.learned_class

    def get_words_count(self):
        return self.words_count

# Returns a dictionary that maps each word to its frequency in the given text
def sum_of_pairs(text):
    word_count = {}
    for pair in text:
        word  = pair[0]
        count = pair[1]
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += float(count)
    return word_count

# Saves data to pkl file
# @param name: Name of the file to be saved in
# @param data: Data to be saved
def save_data(dir_path, name, data):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(os.path.join(dir_path, name + '.pkl'), 'wb') as f:
        pickle.dump(data, f)

# Loads the data from file
# @param name: Name of the file to load from
def load_data(dir_path, name):
    with open(os.path.join(dir_path, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

# This function creates dictionaries that store information about the dataset
# @param file_name: input file name
# @param save_file_name: name of the data file to be saved
# @return N_d: The number of documents
# @return data_set: Dictionary from document name to its instance
# @return class_count: Dictionary from class to its frequency
# @return class_text: Dictionary from class c to list of pairs <word, count> of
# documents of class c
#
# Complexity: O(|D| x L) where |D| is the number of documents and L is the avg
# length of a document
def make_data_set(dataset_name, file_name, save_file_name):
    dir_path = dataset_name + '_data'
    class_count = {}
    class_text  = {}
    if os.path.exists(os.path.join(dir_path, save_file_name + '.pkl')):
        return load_data(dir_path, save_file_name)
    else:
        data_set = {}
        with open(file_name, 'r') as f:
            lines = f.readlines()
            N_d = len(lines) # total number of documents
            for line in lines:
                tokens = line.split()
                doc_name = tokens[0]
                _class = tokens[1]
                if _class not in class_count:
                    class_count[_class] = 1
                else:
                    class_count[_class] += 1
                words_list = []
                for i in range(2, len(tokens), 2):
                    word  = tokens[i]
                    count = tokens[i+1]
                    words_list.append((word, count))
                if _class not in class_text:
                    class_text[_class] = words_list
                else:
                    class_text[_class] += words_list
                data_set.update({doc_name: Document(_class, sum_of_pairs(words_list))})
        save_data(dir_path, save_file_name, [N_d, data_set, class_count, class_text])
        return N_d, data_set, class_count, class_text

# Train the multinomial model
# @param V: List of vocabulary
# @return prior_prob: Prior probabilites of each class
# @return cond_prob: Conditional probability of each word in V given a class c
#
# Complexity: O(|C|x|V|) where |C| is the number of classes and |V| is the size
# of the vocabulary
def train(dataset_name, training_file_name, V, smoothing):
    prior_prob = {}
    cond_prob  = {}

    data = make_data_set(dataset_name, training_file_name, 'training_data')

    N_d = data[0]
    class_count = data[2]
    class_text = data[3]

    for c in classes:
        cnt_c  = class_count[c]
        text_c = class_text[c]
        prior_prob[c] = np.log(cnt_c/N_d)
        N_w = sum_of_pairs(text_c)
        N = len(text_c) # Number of all words of documents of class c
        for voc_word in V:
            if voc_word not in N_w: N_w[voc_word] = 0
            if smoothing:
                cond_prob[(voc_word, c)] = np.log((N_w[voc_word] + 1)/(N + len(V)))
            else:
                if N_w[voc_word] == 0:
                    cond_prob[(voc_word, c)] = 0
                else:
                    cond_prob[(voc_word, c)] = np.log(N_w[voc_word]/N)
    return prior_prob, cond_prob

# Test the multinomial model
# @param doc: Document instance
# @param prior_prob: prior probabilies
# @param cond_prob: Conditional probabilites
# @param vocabs: List of vocabulary
# @param n: Number of first entries of vocabs
#
# Complexity: O(|C| x n) where |C| is the number of classes and L is the number
# of words of the documents that are in the first n entries of the vocabs
def test(doc, prior_prob, cond_prob, vocabs, n):
    words_to_check = vocabs[:n]

    words = []
    for word, count in doc.get_words_count().items():
        if word in words_to_check:
            words.append(word)

    # log probabilities
    class_score = {}
    for c in classes:
        class_score[c] = prior_prob[c]
        for word in words:
            if (word, c) in cond_prob:
                class_score[c] += cond_prob[(word, c)]

    learned_class = ''
    max = -1000000000
    for c in classes:
        if class_score[c] > max:
            max = class_score[c]
            learned_class = c
    return learned_class

# Calculates the error rate
# @param N_t: Number of documents in test dataset
# @param n: Number of first entries of vocabs
#
# Complexity: O(|D| x Complexity(test))
def cal_error(N_t, test_data_set, prior_prob, cond_prob, vocabs, n):
    correct_label = 0
    for doc_name, doc in test_data_set.items():
        doc.set_learned_class(test(doc, prior_prob, cond_prob, vocabs, n))
        if (doc.get_true_class() == doc.get_learned_class()):
            correct_label += 1
    return 1 - correct_label/N_t

# Plots the error rates as function of vocabulary size
# @param N_t: Number of documents in test dataset
# @param limit: max number of first entries of vocab list
def my_plot(N_t, test_data_set, prior_prob, cond_prob, vocabs):
    error_list = []
    for n in test_sizes:
        error_list.append(cal_error(N_t, test_data_set, prior_prob, cond_prob, vocabs, n))
    logged_test_sizes = np.log(test_sizes)
    plt.xlabel("log(Voc Size)")
    plt.ylabel("Error rate")
    plt.plot(logged_test_sizes, error_list, 'ro', linestyle='dashed')
    plt.show()

# Generates the Confusion matrix
# @return df_conf_norm: Normalized confusion matrix
def gen_confusion_matrix(test_data_set):
    actual_labels = []
    predicted_labels = []
    for doc_name, doc in test_data_set.items():
        actual_labels.append(doc.get_true_class())
        predicted_labels.append(doc.get_learned_class())

    y_actual = pd.Series(actual_labels, name='Actual')
    y_predicted = pd.Series(predicted_labels, name='Predicted')
    df_confusion = pd.crosstab(y_actual, y_predicted)
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    return df_conf_norm

# Plots the confusion matrix
def plot_confusion_matrix(df_confusion, cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=90)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()

# Runs the model on the 20news dataset
def apply_on_dataset(dataset_name, smoothing):
    training_data = os.path.join(dataset_name, dataset_name + '.tr')
    voc_data = os.path.join('20news', '20news.voc')
    test_data = os.path.join(dataset_name, dataset_name + '.te')

    # Read vocabs and store them in a list
    with open(voc_data, 'r') as f:
        vocabs = [word for line in f for word in line.split()]

    print('Training...')
    prior_prob, cond_prob = train(dataset_name, training_data, vocabs, smoothing)

    print('Reading Test Dataset')
    data = make_data_set(dataset_name, test_data, 'testing_data')
    N_t = data[0]
    test_data_set = data[1]

    print('Testing and Plotting...')
    my_plot(N_t, test_data_set, prior_prob, cond_prob, vocabs)

    plot_confusion_matrix(gen_confusion_matrix(test_data_set))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('Dataset_Name', help='Name of the dataset (20news or spam)')
    args = parser.parse_args()

    data_set_name = sys.argv[1]
    data_set_name = data_set_name.strip() # removes extra spaces

    if data_set_name.lower() != 'spam' and data_set_name.lower() != '20news':
        sys.exit('Please enter a valid dataset name (20news or spam)')

    if data_set_name == '20news':
        classes = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
        'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
        'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
        'soc.religion.christian','talk.politics.guns','talk.politics.mideast',
        'talk.politics.misc','talk.religion.misc']
    elif data_set_name == 'spam':
        classes = ['spam', 'ham']

    apply_on_dataset(data_set_name, smoothing=True)
