import numpy as np
from typing import Text, List, Any, Optional

class NaiveBayes:

  def __init__(self):
    self.vocabulary = set() 
    self.class_priors = {}
    self.word_counts_for_classes = {}
    self.word_freq = {}
  def fit(self, training_corpus: List[List[List[Text],Text]]) -> None:
    """This function takes the trainig corpus and calculates the prior probabilities of each class 
    and counts the occurrence of every word in every possible class.

    ...

    Parameters
    ----------
    training_corpus : List[List[List[Text],Text]]
        training corpus is list of trainng examples. 
        Each example is [sentence,class]
        sentence is a list of words.
    """
    class_count = {}
    for doc,cls in training_corpus: 
      self.word_freq[cls] = self.word_freq.get(cls,0)+len(doc)
      class_count[cls] = class_count.get(cls,0)+1
      word_dict = self.word_counts_for_classes.get(cls,{})
      for word in doc:
        self.vocabulary.add(word)
        word_dict[word] = word_dict.get(word,0)+1 
      self.word_counts_for_classes[cls] = word_dict 
    for cls in class_count:
      self.class_priors[cls] = np.log(class_count[cls]/len(training_corpus)) 
    print("Done!!!")

  def predict_class(self, test_sample : Text, smoothing: Text = None) -> Text:
    """Predicts and returns the class of a document based on the prior probablities of class,likelihood of words.

    ...

    Parameters
    ----------
    training_sample : Text
        testing sentence.
    """
    x = y = 0
    if(smoothing == "laplace"):
      x = 1
      y = len(self.vocabulary) 
    posteriors = {}
    for cls in self.class_priors:
      likelihood = 0 
      for word in test_sample:
        likelihood += np.log((self.word_counts_for_classes[cls].get(word,0)+x)/(self.word_freq[cls]+y)) 
      posteriors[cls] = likelihood+self.class_priors[cls] 
    return max(posteriors,key = lambda x:posteriors[x])