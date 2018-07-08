Implementation of a Mutlinomial Text Classifier.

## Description

1. First we read the data from the input files given and store them in pkl files so that we can load them directly when we run the program another time. Then, we loop over each document and process its information and store them in dictionaries. 
Complexity: O(|D| x L) where |D| is the number of documents and L is the average lenght of a document.

2. To train the model, we need to calculate the estimated prior probabilites and conditional probabilites. To estimate the prior probability, we use the relative frequency of documents of each class c w.r.t the number of documents. For the conditional probability, we loop over the words in the vocabulary list and check if it exists in the text of documents of the corresponding class c. If not then the conditional probability is 0 else its the count of this word divided by the total number of words. Later, we addded smoothing to prevent having conditional probabilities with zero value.
Complexity: O(|C| x |V|) where |C| is the number of classes and |V| is the size of the vocabulary.

3. For testing, for each document, we calculate its expected class and compare it with the true class. We used the first n entries of the vocabs words to calculate the probability score for each class. After that, we return the class with the maximum score.	
Complexity: O(|C| x L') where L' is the number of words of the documents that are in the first n entries of the vocabs.

## Smoothing

The model uses Add-One smoothing.

## Results
We test the model by selecting the first k entries from the vocabulary where k is between 1 and n. Then plot the error rates logirathmically. Below are the results after training the model using all the vocabulary words. 
The test is done on the sizes = [500, 1000, 5000, 10000, 20000, 93508]

#### 20news dataset results:
<img width="400" alt="20news_result" src=https://user-images.githubusercontent.com/17355283/42418237-47ce03ec-829c-11e8-9e19-235447f62d65.png> <img width="400" alt="20news_cov_matrix" src=https://user-images.githubusercontent.com/17355283/42418271-d14bdc66-829c-11e8-9d41-4f92ec874518.png>

#### spam dataset results:
<img width="400" alt="spam_result" src=https://user-images.githubusercontent.com/17355283/42418263-b2c1b662-829c-11e8-8071-0d2a9f61e65a.png> <img width="400" alt="spam_cov_matrix" src=https://user-images.githubusercontent.com/17355283/42418268-c37ace8a-829c-11e8-8cb5-29713c631f39.png>
