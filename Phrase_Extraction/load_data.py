# Loading data

source_path = './aligned_corpus/source'
target_path = './aligned_corpus/target'
alignment_path = './aligned_corpus/alignment'

def get_alignments(N):
    """
    This method stores the alignments given in the data files
    
    :param N: The number of sentences
    :return: List of pairs of index alignment between source and target sentences
    """

    res = [[] for _ in range(N)] # 0-29
    with open(alignment_path, 'r') as f:
        sent_num = 0
        for line in f:
            tokens = line.strip().split()
            if not tokens: continue
            if tokens[0] == 'SENT:':
                # 0 based indexing
                sent_num = int(tokens[1])-1
            else:
                # append (e, f) alignment pair
                res[sent_num].append((int(tokens[2]), int(tokens[1])))
    return res

def get_parallel_corpus():
    """
    This methods store the sentences pairs given in the data files

    :return: List of parallel source and target sentences
    """

    res = []
    with open(source_path, 'r') as f_source, open(target_path, 'r') as f_target:
        for f_sent, e_sent in zip(f_source, f_target):
            res.append((f_sent.strip().split(), e_sent.strip().split()))
    return res

