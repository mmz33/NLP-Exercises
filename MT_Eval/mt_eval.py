import string
import argparse
from collections import Counter

class MTEval(object):
    """
    Class for evaluation of Machine Translation based on WER and PER measurements.
    Reference: http://www.statmt.org/wmt07/pdf/WMT07.pdf
    """

    hyp_path = './ex2_data/hyp'  # hypothesis text path
    ref_path = './ex2_data/ref'  # reference text path

    @staticmethod
    def edit_distace(hyp, ref):
        """
        Edit distance algorithm using Dynamic Programming

        dp[i][j] = min distance to transform hyp[1,...i] to ref[1,...j]

        :param hyp: list of the words of the hypothesis
        :param ref: list of the words of the reference
        :return: the minimum distance to transform the hypothesis sentence to reference sentence
        """

        n = len(hyp)
        m = len(ref)

        dp = [[0 for j in range(m+1)] for i in range(n+1)]

        # i deletions
        for i in range(n+1):
            dp[i][0] = i

        # j insertions
        for j in range(m+1):
            dp[0][j] = j

        for i in range(1, n+1):
            for j in range(1, m+1):
                if hyp[i-1] == ref[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

        return dp[n][m]

    @staticmethod
    def per(hyp_words_list, ref_words_list):
        """
        :param hyp_words_list: hypothesis words list
        :param ref_words_list: reference words list
        :return: per for the given hyp and ref sentences
        """
        hyp_count = Counter(hyp_words_list)
        ref_count = Counter(ref_words_list)

        total_sum = 0
        for w in hyp_words_list:
            total_sum += abs(ref_count[w] - hyp_count[w])

        return 0.5 * (abs(len(ref_words_list) - len(hyp_words_list)) + total_sum)

    def eval(self, without_punct=False):
        """
        Evaluates the MT using WER and PER measurements.

        :param without_punct: If true, then punctuations are removed from the sentences.
        :return WER and PER
        """

        total_dist = 0 # total distance computed from edit_distance method
        num_of_ref_words = 0 # total number of words in the reference
        diff = 0 # total difference between the hypothesis and reference sentences used to compute PER

        with open(self.hyp_path, 'r') as hyp_file, open(self.ref_path, 'r') as ref_file:
            for hyp_sent, ref_sent in zip(hyp_file, ref_file):
                hyp_words_list = hyp_sent.split() # hypothesis sentence as list of words
                ref_words_list = ref_sent.split() # reference sentence as list of words

                # removes punctuations
                if without_punct:
                    hyp_words_list = [word for word in hyp_words_list if word not in string.punctuation]
                    ref_words_list = [word for word in ref_words_list if word not in string.punctuation]

                # WER
                total_dist += self.edit_distace(hyp_words_list, ref_words_list)

                # PER
                diff += self.per(hyp_words_list, ref_words_list)

                num_of_ref_words += len(ref_words_list)

        _wer = (total_dist/num_of_ref_words) * 100
        _per = (diff/num_of_ref_words) * 100

        return _wer, _per

if __name__ == '__main__':
    mt_eval = MTEval()

    # With punctuation results
    wer_with_punct, per_with_punct = mt_eval.eval()
    print('wer (with punctuation): ' + '%.3f' % wer_with_punct + ' %')
    print('per (with punctuation): ' + '%.3f' % per_with_punct + ' %')

    print()

    # Without punctuation results
    wer_without_punct, per_without_punct = mt_eval.eval(without_punct=True)
    print('wer (without punctuation): ' + '%.3f' % wer_without_punct + ' %')
    print('per (without punctuation): ' + '%.3f' % per_without_punct + ' %')
