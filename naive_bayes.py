# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
from collections import defaultdict
from math import log
import sys
from pprint import pprint

class NaiveBayes(Classifier):
    u"""A naïve Bayes classifier."""

    def __init__(self, model=defaultdict(dict)):
        super(NaiveBayes, self).__init__(model)

    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model
    model = property(get_model, set_model)

    def train(self, observations, alpha=0.001):
        self.seen = defaultdict(dict)
        total_obs = 0.0
        class_counts = {}
        feature_set = set()
        num_features = 0

        # get counts
        for ob in observations:
            total_obs += 1.0
            class_counts[ob.label] = class_counts.get(ob.label, 0.0) + 1.0
            for feature in list(set(ob.features())):
                self.seen[ob.label][feature] = self.seen[ob.label].get(feature, 0.0) + 1.0
                if feature not in feature_set:
                    num_features += 1
                    feature_set.add(feature)

        # cache the priors in self.seen for use in classifying documents
        for label in self.seen:
            self.seen[label]['PRIOR PROBABILITY'] = class_counts[label] / total_obs

        #calculate posterior probabilities using Laplace smoothing
        for label in self.seen:
            for feature in self.seen[label]:
                self.seen[label][feature] = self.seen[label].get(feature, 0.0) + alpha
                self.seen[label][feature] /= (class_counts[label] + alpha * num_features)

    def classify(self, doc):

        best_label = "Error"
        max_prob = -1*sys.maxsize
        doc_features = set(doc.features())

        for label in self.seen:
            current_prob = 0.0
            label_features = {feature for feature in self.seen[label] if feature != 'PRIOR PROBABILITY'}
            label_feat_num = len(list(label_features))

            # features that occur in the document but aren't in the model
            doc_only = doc_features - label_features
            for feature in list(doc_only):
                current_prob += log(1.0/label_feat_num)

            # features that occur in both the document and the model
            doc_label = doc_features & label_features
            for feature in list(doc_label):
                current_prob += log(self.seen[label][feature])

            # features that occur in the model but not in the document
            label_only = label_features - doc_features
            for feature in list(label_only):
                current_prob += log(1.0-self.seen[label][feature])

            current_prob += log(self.seen[label]['PRIOR PROBABILITY'])
            if current_prob > max_prob:
                max_prob = current_prob
                best_label = label

        return best_label

