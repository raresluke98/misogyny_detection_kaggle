# Kaggle Misogyny Detection

I have scripted an italian tweet misogyny detection program for a project in my Artificial Intelligence course.

## Data pre-processing
The following steps have been taken from the Kaggle user Rageeni Sah<sup>(1)</sup>
* The number of entires for each class has been plotted to check if the dataset is balanced.
* WordCloud images have been generated to help ignore the most common words in a later step.

### Data cleansing
The following steps have been taken:
* Removal of punctuation
* Text tokenization
* Removal of stopwords
* Text stemming

### Cropus
The corpus, which is the final product of all the data pre-processing steps applied on the raw text, is used for obtaining the vocabulary.
Afterwards, the "word to index" dictionary is constructed to ultimately obtain the bag of words model.

## Submission no. 1
This submision uses the Multinomial Naive Bayes classifier.
### Naive Bayes
The Naive Bayes methods use the Bayes theroem with strong independence assumptions about the features<sup>(2)</sup>.

Bayes theorem: P(c|X) = P(X|c) x P(c) / P(X)

### Score
The kaggle f1 score on 40% of the data is 0.72351

## References
1.
