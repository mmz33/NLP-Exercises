from Phrase_Extraction.load_data import get_parallel_corpus, get_alignments

class PhraseExt:
    parallel_corpus = None
    alignments = None
    f_alignment = None

    def __init__(self):
        self.N = 30
        self.parallel_corpus = get_parallel_corpus()
        self.alignments = get_alignments(self.N)

    def get_phrase_pairs(self, alignment, f_sent, e_sent):
        bp = set()
        # Loop over all substrings of the english sentence
        for e_start in range(0, len(e_sent)):
            for e_end in range(e_start, len(e_sent)):

                # Find the minimal match
                f_start = len(f_sent)-1
                f_end = -1
                for e, f in alignment:
                    if e_start <= e <= e_end:
                        f_start = min(f, f_start)
                        f_end = max(f, f_end)

                phrases = self.extract(alignment, f_start, f_end, e_start, e_end, f_sent, e_sent)

                if phrases:
                    bp.update(phrases)
        return bp

    @staticmethod
    def extract(alignment, f_start, f_end, e_start, e_end, f_sent, e_sent):
        phrases = set()

        # Check if there is at least one alignment
        if f_end == -1: return phrases

        # Check if the minimal match is valid
        for e, f in alignment:
            if (f_start <= f <= f_end) and (e < e_start or e > e_end):
                return phrases

        f_alignment = [f for _, f in alignment]

        f_s = f_start
        while True:
            f_e = f_end
            while True:
                src_sent = ' '.join(f_sent[i] for i in range(f_s, f_e+1))
                target_sent = ' '.join(e_sent[i] for i in range(e_start, e_end+1))
                phrases.add(((e_end - e_start + 1), src_sent, target_sent))
                f_e += 1
                if f_e in f_alignment or f_e == len(f_sent):
                    break
            f_s -= 1
            if f_s in f_alignment or f_s < 0:
                break

        return phrases

    @staticmethod
    def order_by_length(phrases_set):
        return sorted(phrases_set, key=lambda x: x[0])

    def simple_test(self):
        e_sent = "michael assumes that he will stay in the house"
        f_sent = "michael geht davon aus , dass er im haus bleibt"
        alignment = [(0, 0), (1, 1), (1, 2), (1, 3), (2, 5), (3, 6), (4, 9), (5, 9), (6, 7), (7, 7), (8, 8)]

        res = self.order_by_length(self.get_phrase_pairs(alignment, f_sent.split(), e_sent.split()))

        self.formatted_print(res)

    @staticmethod
    def formatted_print(phrases_set):
        for i, (_, a, b) in enumerate(phrases_set):
            print('({x}) {f_sent} - {e_sent}'.format(x=i+1, f_sent=a, e_sent=b))

if __name__ == '__main__':
    phrase_ext = PhraseExt()

    phrase_ext.simple_test()



