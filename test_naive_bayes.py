# -*- mode: Python; coding: utf-8 -*-

from __future__ import division

from corpus import Document, BlogsCorpus, NamesCorpus
from naive_bayes import NaiveBayes

import sys
from random import shuffle, seed
from unittest import TestCase, main, skip

import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

class EvenOdd(Document):
    def features(self):
        """Is the data even or odd?"""
        return [self.data % 2 == 0]

class BagOfWords(Document):
    def features(self):
        """Trivially tokenized words."""
        return self.data.split()

class Tokenized(Document):
    def features(self):
        """Tokenize using NLTK and remove stopwords & punctuation"""
        stop = set(stopwords.words('english'))
        for word in string.punctuation:
            stop.add(word)
        return [word.lower() for word in word_tokenize(self.data) if word not in stop]

class Bigrams(Document):
    def features(self):
        """Trivially split, then use bigrams"""
        words = self.data.split()
        return [(words[i], words[i+1]) for i in range(len(words)-1)]

class Trigrams(Document):
    def features(self):
        """Trivially split, then use trigrams"""
        words = self.data.split()
        return [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]

class Name(Document):
    def features(self, letters="abcdefghijklmnopqrstuvwxyz"):
        """From NLTK's names_demo_features: first & last letters, how many
        of each letter, and which letters appear."""
        name = self.data
        return ([name[0].lower(), name[-1].lower()] +
                [name.lower().count(letter) for letter in letters] +
                [letter in name.lower() for letter in letters])

def accuracy(classifier, test, verbose=sys.stderr):
    correct = [classifier.classify(x) == x.label for x in test]
    if verbose:
        #print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
        print("%.2d%% " % (100 * sum(correct) / len(correct)), file=verbose)
    return sum(correct) / len(correct)

def fmeasure(classifier, test):

    fmeasures = {}
    results = Counter([(x.label, classifier.classify(x)) for x in test])
    model = classifier.get_model()
    all_labels = model.keys()

    for label in model:
        other = list(all_labels - label)[0]
        tp = results[(label, label)]
        fp = results[(other, label)]
        fn = results[(label, other)]
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)

        # to ensure you don't divide by 0
        if precision == 0 or recall == 0:
            fmeasures[label] = 0.0
        else:
            fmeasures[label] = (2/(1/precision + 1/recall))
    return sum([fmeasures[label] for label in model])/len(list(all_labels))

class NaiveBayesTest(TestCase):
    u"""Tests for the naïve Bayes classifier."""

    def test_even_odd(self):
        """Classify numbers as even or odd"""
        classifier = NaiveBayes()
        classifier.train([EvenOdd(0, True), EvenOdd(1, False)])
        test = [EvenOdd(i, i % 2 == 0) for i in range(2, 500)]
        self.assertEqual(accuracy(classifier, test), 1.0)

    def split_names_corpus(self, document_class=Name):
        """Split the names corpus into training and test sets"""
        names = NamesCorpus(document_class=document_class)
        self.assertEqual(len(names), 5001 + 2943) # see names/README

        seed(hash("names"))
        shuffle(names)
        return (names[:100], names[100:200])

    def test_names_nltk(self):
        """Classify names using NLTK features"""
        train, test = self.split_names_corpus()
        classifier = NaiveBayes()
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.70)

    def split_blogs_corpus(self, document_class):
        """Split the blog post corpus into training and test sets"""
        blogs = BlogsCorpus(document_class=document_class)
        self.assertEqual(len(blogs), 3232)
        seed(hash("blogs"))
        shuffle(blogs)
        return (blogs[:3000], blogs[3000:])

    def test_blogs_bag(self):
        """Classify blog authors using bag-of-words"""
        train, test = self.split_blogs_corpus(BagOfWords)
        classifier = NaiveBayes()
        classifier.train(train)
        classifier.save('alpha01_bagofwords.pkl')
        print('F measure', fmeasure(classifier, test))
        self.assertGreater(accuracy(classifier, test), 0.55)

    def test_blogs_tokens(self):
        train, test = self.split_blogs_corpus(Tokenized)
        classifier = NaiveBayes()
        classifier.train(train)
        classifier.save('alpha01_tokens.pkl')
        print('F measure', fmeasure(classifier, test))
        # classifier.save('tokens.pkl')
        self.assertGreater(accuracy(classifier, test), 0.55)

    def test_blogs_bigrams_01(self):
        train, test = self.split_blogs_corpus(Bigrams)
        classifier = NaiveBayes()
        classifier.train(train)
        # classifier.load('bigrams.pkl')
        print('F measure', fmeasure(classifier, test))
        # classifier.save('bigrams.pkl')
        self.assertGreater(accuracy(classifier, test), 0.55)

    def test_blogs_trigrams(self):
        train, test = self.split_blogs_corpus(Trigrams)
        classifier = NaiveBayes()
        classifier.train(train)
        # classifier.load('trigrams.pkl')
        print('F measure', fmeasure(classifier, test))
        # classifier.save('trigrams.pkl')
        self.assertGreater(accuracy(classifier, test), 0.55)

    def split_blogs_corpus_imba(self, document_class):
        blogs = BlogsCorpus(document_class=document_class)
        imba_blogs = blogs.split_imbalance()
        return (imba_blogs[:1600], imba_blogs[1600:])

    def test_blogs_imba(self):
        train, test = self.split_blogs_corpus_imba(BagOfWords)
        classifier = NaiveBayes()
        classifier.train(train)
        # classifier.save('alpha01_imbalanced.pkl')
        print(fmeasure(classifier, test))
        # you don't need to pass this test
        self.assertGreater(accuracy(classifier, test), 0.1)

if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)
