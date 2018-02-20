<b>Overview</b>

This repo is a basic implementation of a Naive Bayes classifier using numpy arrays. I tested the classifier's performance on the Blog Gender Set from EMNLP 2010. For more information, see: https://www.cs.uic.edu/~liub/publications/EMNLP-2010-blog-gender.pdf

This Naïve Bayes classifier is constructed using Lidstone smoothing. With a basic bag-of-words approach to features and a smoothing factor alpha of 0.01, it achieved 66% accuracy. With that as a baseline, I tested out other features in an attempt to improve the classifier’s performance.

<b>Tokenization</b>

In addition to trivially tokenizing the text (as is done in the pre-built BagofWords document class), I tested what would happen if the text was tokenized more intelligently and if stop words were removed. To do this, I implemented a separate Tokenized Document class. The feature method of this class considers stop words to be all punctuation, plus words in NLTK’s stop words corpus. It tokenizes the text using NLTK’s Penn Treebank word tokenizer and then removes stopwords. Tokenization did not significantly improve results. In fact, the non-tokenized model performed slightly better than the tokenized model, though the differences in performance between the two were really negligible.

<b>Smoothing Factor</b>

I changed the smoothing factor alpha, using factors between 1.0 and 0.001. The best results were achieved with a smoothing factor of 0.1. Factors at both ends of the tested range (1.0 and 0.001) produced poor results on the balanced data set but had high accuracy on the imbalanced data set. Models with extreme learning rates had low F-scores on the imbalanced data set, though, suggesting that they achieved such high accuracy by highly favoring one class over the other.

<b>N-grams</b>

I tried bigrams and trigrams with smoothing factors of 0.01 and 0.001. The bigram and trigrams models took considerably longer to train and test that the unigram models did, likely because there are many, many more different bigrams and trigrams in the data than there are unigrams. I expected the n-gram models to outperform the simple tokenized and bag of words models, because the n gram models capture information about word co-occurrence. Interestingly, while the bigram models outperformed the trigram models, neither one beat the performance of a simple bag of words.
