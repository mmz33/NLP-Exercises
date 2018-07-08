import collections
import time

class Ibm1Model:
    """
    Class that represents the IBM-1 Model for statistical machine translation
    """

    __NUMBER_OF_FREQ_WORDS__ = 1000 # Most frequent words number
    __NUMBER_OF_PARALLEL_SENT__ = 100000 # Number of training samples (MAX = 270149)

    english_text_path = './ex4_data/news-commentary-v12.de-en.tok.en' # english text path
    deutsch_text_path = './ex4_data/news-commentary-v12.de-en.tok.de' # deutsch text path

    eng_de_pairs = [] # list of (english, deutsch) sentences pairs
    number_of_eng_words = 0 # Number of english words
    number_of_de_words = 0 # Number of de words

    alignment_prob = collections.defaultdict(float)
    estimated_lexicon_prob = collections.defaultdict(float) # p(de_j | e_i)

    eng_count = collections.Counter()  # freq counter for english words
    de_count = collections.Counter()  # freq counter for deutsch words

    en_index = {} # index dict for english words
    de_index = {} # index dict for de words

    en_word_to_index = {}
    de_word_to_index = {}

    en_sent_index_map = {}
    de_sent_index_map = {}

    def read_input(self):
        """
        Reads the input texts (English and Deutsch corpuses)
        """
        print('Reading input...')
        limit = 0
        with open(self.english_text_path, 'r') as e_file, open(self.deutsch_text_path) as de_file:
            for e_sent, de_sent in zip(e_file, de_file):

                if limit >= self.__NUMBER_OF_PARALLEL_SENT__: break

                e_words = e_sent.strip().split()
                de_words = de_sent.strip().split()

                if len(e_words) == 0 or len(de_words) == 0: continue

                self.eng_de_pairs.append((e_words, de_words))

                for e_word in e_words:
                    self.eng_count[e_word] += 1

                for de_word in de_words:
                    self.de_count[de_word] += 1
                    self.alignment_prob[de_word] = 1/len(e_words)

                limit += 1

        self.number_of_eng_words = len(self.eng_count.keys())
        self.number_of_de_words  = len(self.de_count.keys())

        print('Number of training samples:', len(self.eng_de_pairs))
        print('Number of english words:', self.number_of_eng_words)
        print('Number of de words:', self.number_of_de_words)

    def index_most_freq_words(self):
        """
        Index the most frequent k words
        Index the parallel training samples sentences
        Map each sent index to a list of words indexes

        This indexing is done for speedup.
        """

        most_freq_eng = self.eng_count.most_common(self.__NUMBER_OF_FREQ_WORDS__)

        self.en_index[0] = '<UNK>'
        self.en_word_to_index['<UNK>'] = 0

        self.de_index[0] = '<UNK>'
        self.de_word_to_index['<UNK>'] = 0

        index = 1
        for e,_ in most_freq_eng:
            self.en_index[index] = e
            self.en_word_to_index[e] = index
            index += 1

        most_freq_de = self.de_count.most_common(self.__NUMBER_OF_FREQ_WORDS__)
        index = 1
        for de,_ in most_freq_de:
            self.de_index[index] = de
            self.de_word_to_index[de] = index
            index += 1

        self.__NUMBER_OF_FREQ_WORDS__ = min(self.__NUMBER_OF_FREQ_WORDS__, len(most_freq_eng), len(most_freq_de))

        unk_cnt = 0

        sent_index = 0
        for (en_sent, de_sent) in self.eng_de_pairs:

            if sent_index >= self.__NUMBER_OF_PARALLEL_SENT__: break

            en_index_list = []
            for e in en_sent:
                if e not in self.en_word_to_index:
                   en_index_list.append(0) # <UNK>
                else:
                    en_index_list.append(self.en_word_to_index[e])

            self.en_sent_index_map[sent_index] = en_index_list

            de_index_list = []
            for de in de_sent:
                if de not in self.de_word_to_index:
                    de_index_list.append(0) # <UNK>
                    unk_cnt += 1
                else:
                    de_index_list.append(self.de_word_to_index[de])

            self.de_sent_index_map[sent_index] = de_index_list

            sent_index += 1

        self.de_count['<UNK>'] = unk_cnt

    def em_algorithm(self, iterations=10):
        """
            Trains the IBM1 model to learn the lexicon probabilities parameters

            :param iterations: Number of EM algorithm iterations
        """

        self.read_input() # Reads input files

        print('Indexing...')
        self.index_most_freq_words()

        total_time_start = time.time()

        print('Training IBM-1 Model...')

        print('Initializing lexicon model uniformaly distributed...')

        # Initialize p(de|e)
        start = time.time()

        for n_sent in range(self.__NUMBER_OF_PARALLEL_SENT__):

            for i in self.en_sent_index_map[n_sent]:
                e_word = self.en_index[i]
                for j in self.de_sent_index_map[n_sent]:
                    de_word = self.de_index[j]
                    self.estimated_lexicon_prob[(e_word, de_word)] = self.de_count[de_word]/self.number_of_de_words

        end = time.time()
        print('Time for initializing:', end-start)

        # Initializes normalization to 0
        e_norm = collections.defaultdict(float)

        for iteration in range(iterations):
            print('Running EM iteration {n}...'.format(n=(iteration+1)))

            start = time.time()

            # Initializes the counts to 0
            e_de_count = collections.defaultdict(float)
            e_count = collections.defaultdict(float)

            for n_sent in range(self.__NUMBER_OF_PARALLEL_SENT__):

                # Computes the normalization
                for i in self.en_sent_index_map[n_sent]:
                    e_word = self.en_index[i]
                    e_norm[e_word] = 0.0
                    for j in self.de_sent_index_map[n_sent]:
                        de_word = self.de_index[j]
                        e_norm[e_word] += self.estimated_lexicon_prob[(e_word, de_word)]

                # Computes the counts
                for i in self.en_sent_index_map[n_sent]:
                    e_word = self.en_index[i]
                    for j in self.de_sent_index_map[n_sent]:
                        de_word = self.de_index[j]
                        lexicon_prob = self.estimated_lexicon_prob[(e_word, de_word)]/e_norm[e_word]
                        e_de_count[(e_word, de_word)] += lexicon_prob
                        e_count[e_word] += lexicon_prob

                end = time.time()

            print('Time for iteration {n}: {t}'.format(n=iteration+1, t=end-start))

            print('Reestimating the probabilities...')

            start = time.time()

            for i in range(self.__NUMBER_OF_FREQ_WORDS__):
                e_word = self.en_index[i]
                for j in range(self.__NUMBER_OF_FREQ_WORDS__):
                    de_word = self.de_index[j]
                    v = (e_word, de_word)
                    self.estimated_lexicon_prob[v] = e_de_count[v] / e_count[e_word]

            end = time.time()

            print('Time for reestimating:', end-start)

        total_time_end = time.time()

        print('Total time:', total_time_end - total_time_start)

    def get_result(self):
        self.em_algorithm(iterations=3)

        print('Computing the results...')

        # list of de words
        de_words = self.de_count.keys()

        res = {}

        for e_word, _ in self.eng_count.most_common(30):
            prob = collections.Counter()
            for de_word in self.de_index.values():
                prob[de_word] = self.alignment_prob[de_word] * self.estimated_lexicon_prob[(e_word, de_word)]

            res[e_word] = [(word, p) for word, p in prob.most_common(3)]
            print(e_word + ': {}'.format(res[e_word]))

if __name__ == '__main__':
    model = Ibm1Model()
    model.get_result()
