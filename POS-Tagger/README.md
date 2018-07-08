# POS-Tagger

Implementation of a bigram (Hidden Markov Model) based POS tagger. It uses Viterbi algorithm which uses dynamic programming to find the best sequence of tags for a given sentence.
The model is tested on Wall Street Journal corpus.

The objective function that we want to maximize:
<img width="577" alt="obj_function" src="https://user-images.githubusercontent.com/17355283/41366742-592af0fe-6f3d-11e8-9cb4-94c3ec60a3bf.png">

where ![g](https://latex.codecogs.com/gif.latex?g_%7B1%7D%5E%7BN%7D) is the sequence of tags and ![w](https://latex.codecogs.com/gif.latex?w_%7B1%7D%5E%7BN%7D) is the sequence of given words.

![p0](https://latex.codecogs.com/gif.latex?p_0) means zero-order dependence (depends only on n)

![p1](https://latex.codecogs.com/gif.latex?p_1) means first order dependence (depends on n and n-1)

### Smoothing
The model uses Turing good estimate (k = 5) for smoothing bigram tags probabilities.

### out-of-vocabulary words (OOV)
For handling OOV, a naive approach would be to assign a low probability such as 0.00001 for unknown words. The model handles OOV based
on the number of word suffixes of fixed sizes that range from 1 to L occuring with each tag in the tags set. For example,
words ending with 'ed' has most probably past tense or past participle tags and so p(w|VBD) would be high.

<img width="588" alt="oov" src="https://user-images.githubusercontent.com/17355283/41366773-6e4c06bc-6f3d-11e8-9df1-ded5ece0d369.png">

### Results
The model achieves 92.7% accuracy on the given dataset.

### References: 
- https://web.stanford.edu/~jurafsky/slp3/10.pdf
- ftp://ftp.nada.kth.se/Theory/Viggo-Kann/tagger.pdf
