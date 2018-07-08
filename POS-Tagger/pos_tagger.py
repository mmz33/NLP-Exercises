import collections
import random

# training datasets paths
training_text_path = 'wsj/wsj.text.tr'
training_tags_path = 'wsj/wsj.pos.tr'

# testing datasets paths
test_text_path = 'wsj/wsj.text.test'
test_tags_path = 'wsj/wsj.pos.test'

# Smoothing model for tags model
class TuringGood:
    k = 5

    def __init__(self, tags, tag_bigram_count, tag_count, num_of_tags):
        self.tags = tags
        self.tag_bigram_count = tag_bigram_count
        self.tag_count = tag_count
        self.num_of_tags = num_of_tags

        self.cofc = collections.defaultdict(int)
        for count in self.tag_bigram_count.values():
            self.cofc[count] += 1

        self.a_h_cache = {}
        self.beta_hash = {}
        self.norm_cache = {}

    # Computes lambda_r discounting parameter
    def lambda_r(self, r):
        x = ((r+1) * self.cofc[r+1])/(r * self.cofc[r])
        y = ((self.k+1) * self.cofc[self.k+1])/self.cofc[1]
        return (1-x)/(1-y)

    # Computes A(h)
    def compute_a_h(self, h):
        if h in self.a_h_cache:
            return self.a_h_cache[h]

        a_h = 0
        for w in self.tags:
            cnt = self.tag_bigram_count[(h, w)]
            if 1 <= cnt <= self.k:
                a_h += self.lambda_r(cnt) * (self.tag_bigram_count[(h, w)]/self.tag_count[h])

        self.a_h_cache[h] = a_h
        return a_h

    # Computes beta(w|h_bar)
    def get_beta(self, w):
        if w in self.beta_hash:
            return self.beta_hash[w]

        res = self.tag_count[w]/self.num_of_tags
        self.beta_hash[w] = res
        return res

    # Computes the normalization for beta distribution
    def get_norm(self, h):
        if h in self.norm_cache:
            return self.norm_cache[h]
        res = sum([self.get_beta(w) for w in self.tags if self.tag_bigram_count[(h, w)] == 0])
        self.norm_cache[h] = res
        return res

    # Returns turing good probability for the given bigram
    def get_prob(self, bigram):
        w = bigram[1]
        h = bigram[0]

        cnt = self.tag_bigram_count[bigram]

        p = 0
        if cnt > self.k:
            p = cnt/self.tag_count[h]
        elif 1 <= cnt <= self.k:
            lam = self.lambda_r(cnt)
            if lam < 1:
                p = (1 - lam) * (cnt/self.tag_count[h])
            else:
                a_h = self.compute_a_h(h)
                beta_w_h = self.get_beta(w)
                normalization = self.get_norm(h)
                p = a_h * (beta_w_h/normalization)

        return p

# Bigram based POS tagger bigram class
class POStagger:
    tag_bigrams = [] # stores list of tag bigrams
    word_tag_pairs = [] # stores list of word tag pairs

    # tags and words lists
    tags = []
    words = []

    # dictionaries to store counts
    words_count = collections.defaultdict(int) # N(w)
    tag_bigram_count = collections.defaultdict(int) # N(g', g)
    tag_count = collections.defaultdict(int) # N(g)
    word_tag_count = collections.defaultdict(int) # (w, g)

    num_of_words = 0 # total number of words
    num_of_tags = 0 # total number of tags

    suffix_prob_cache = collections.defaultdict(int)
    suffix_cnt_cache = collections.defaultdict(int)
    suffix_norm_cache = collections.defaultdict(int)

    # Reads the training dataset
    # Fills the count dictionaries
    def __init__(self):
        with open(training_text_path, 'r') as text_file, open(training_tags_path, 'r') as tag_file:
            for _sentence, _tags in zip(text_file, tag_file):
                words = _sentence.split()
                tags = _tags.split()

                words = [word.lower() for word in words]
                words = ['<S>'] + words

                tags = ['<S>'] + tags

                prev_tag = ''
                for w, t in zip(words, tags):

                    # tag processing
                    tag_bigram = (prev_tag, t)
                    self.tag_bigrams.append(tag_bigram)
                    self.tag_bigram_count[tag_bigram] += 1
                    self.tag_count[t] += 1

                    # word tag processing
                    word_tag = (w, t)
                    self.word_tag_pairs.append(word_tag)
                    self.word_tag_count[word_tag] += 1

                    self.words_count[w] += 1

                    prev_tag = t

        self.num_of_words = len(self.words_count.keys())
        self.num_of_tags = len(self.tag_count.keys())

        self.tags = list(self.tag_count.keys())
        self.words = list(self.words_count.keys())

        self.turing_good = TuringGood(self.tags, self.tag_bigram_count, self.tag_count, self.num_of_tags)

    # p(g) = N(g)/G
    def get_tag_prob(self, tag):
        return self.tag_count[tag]/self.num_of_tags

    # p(g'|g) = N(g',g)/N(g')
    def get_trans_prob(self, bigram, smoothing=False):
        if smoothing:
            return self.turing_good.get_prob(bigram)

        return self.tag_bigram_count[bigram]/self.tag_count[bigram[0]]

    # OOV handling (not used because of bad accuracy)
    @staticmethod
    def get_expected_tag(word):
        # tag = random.choice(['NN', 'JJ'])
        tag = ''
        if list(word)[0].isupper():
            tag = 'NNS'
        elif '-' in word:
            tag = random.choice(['NN', 'JJ'])
        elif word.endswith('ed'):
            tag = random.choice(['VBN', 'VBD'])  # past participle or past tense
        elif word.endswith('ing'):
            tag = 'VBG'  # present participle
        elif word.endswith('er'):
            tag = 'JJR'  # comparative
        elif word.endswith('s'):
            tag = 'NNS'  # plural

        return tag

    # OOV Handling using suffix counting probability
    def get_suffix_prob(self, word, tag, suffix_size=3, alpha=0.00001):
        if len(word) <= 3: return 0.000001

        suffix = word[-suffix_size:]

        if (suffix, tag) in self.suffix_prob_cache:
            return self.suffix_prob_cache[(word, tag)]

        if (suffix, tag) not in self.suffix_cnt_cache:
            total = 0
            for i in range(suffix_size):
                suffix = word[-(i + 1):]
                for tag in self.tags:
                    self.suffix_cnt_cache[(suffix, tag)] += 1
                    total += self.suffix_cnt_cache[(suffix, tag)]
                self.suffix_norm_cache[suffix] = total

        p = 0
        for i in range(suffix_size):
            suffix = word[-(i + 1):]
            p += alpha * ((self.suffix_cnt_cache[(suffix, tag)])/self.suffix_norm_cache[suffix])

        self.suffix_prob_cache[(suffix, tag)] = p
        return p

    # p(w|g) = N(w,g)/N(g)
    def get_emission_prob(self, word_tag, handle_oov=False):
        if handle_oov:
            if self.word_tag_count[word_tag] == 0:
                return self.get_suffix_prob(word_tag[0], word_tag[1])

        return self.word_tag_count[word_tag]/self.tag_count[word_tag[1]]

    # Returns the best tag sequence for the given sentence
    # Q(n,g) = max Q(n-1,g')*p_0(w|g)*p_1(g'|g)
    def get_best_tags(self, sent):
        backpointers = [] # to track the path of tags

        prev_tag = '<S>'
        for i in range(0, len(sent)):
            w = sent[i]
            max_v = -1
            max_tag = ''
            for t in self.tags:
                if t == '<S>': continue
                # p_0(w|g) * p_1(g'|g)
                v = self.get_emission_prob((w, t), handle_oov=True) * self.get_trans_prob((prev_tag, t), smoothing=True)
                if v > max_v:
                    max_v = v
                    max_tag = t

            backpointers.append(max_tag)
            prev_tag = max_tag

        return backpointers

    # Test the model on the test dataset
    # Returns the accuracy in %
    def test(self):
        correct = 0
        total = 0
        with open(test_text_path, 'r') as text_file, open(test_tags_path, 'r') as tag_file:
            for _sentence, _tags in zip(text_file, tag_file):
                words = _sentence.split()
                tags = _tags.split()

                words = [word.lower() for word in words]

                best_tag_seq = self.get_best_tags(words)

                for t1, t2 in zip(tags, best_tag_seq):
                    if t1 == t2:
                        correct += 1
                    total += 1

        return (correct/total)*100

if __name__ == '__main__':
    print('Training model...')
    pos_tagger = POStagger()
    print('Testing model...')
    accuracy = pos_tagger.test()
    print('Accuracy: ' + '%.5f' % accuracy + ' %')
