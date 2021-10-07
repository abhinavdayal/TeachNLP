import numpy as np
from typing import Text, List, Any, Optional

class NaiveBayes:
  """
  A class to implement Naive Bayes Language model for classification tasks.
  ...

  Attributes
  ----------
  vocabulary : set()
    Set of all words in training corpus
  class_priors : dict()
    prior probabilities of all classes. (key : value) <==> (class name : prior probability)
  classes_and_each_words_freq__dict : {class:{word1:freq,word2:freq,.....}}
    This dictionary contains all classes mapped with corresponding word-frequency dictionaries
  classes_word_count_dict : {class : Number of words in all documents}
    This dictionary contains each class mapped with number of words in all documents of that class.
  
  Methods
  -------
  fit(training_corpus: List[List[List[Text],Text]])
    fits the model with given training corpus
  predict_class(test_document : Text, smoothing: Text = None)
    Takes a document(can be sentence or a paragraph) and predicts the class for that document.
  """

  def __init__(self):
    self.vocabulary = set() 
    self.class_priors_dict = dict()
    self.classes_and_each_words_freq__dict = dict()
    self.classes_word_count_dict = dict()


  def fit(self, training_corpus: List) -> None:
    """This function takes the trainig corpus and calculates the prior probabilities of each class 
    and counts the occurrence of every word in every possible class.
    Each training sample is document and corresponding class.
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
      self.classes_word_count_dict[cls] = self.classes_word_count_dict.get(
          cls, 0)+len(doc)
      class_count[cls] = class_count.get(cls,0)+1
      word_dict = self.classes_and_each_words_freq__dict.get(cls, {})
      for word in doc:
        self.vocabulary.add(word)
        word_dict[word] = word_dict.get(word,0)+1 
      self.classes_and_each_words_freq__dict[cls] = word_dict
    for cls in class_count:
      self.class_priors[cls] = np.log(class_count[cls]/len(training_corpus)) 
    print("Done!!!")

  def predict_class(self, test_document: Text, smoothing: Text = None) -> Text:
    """Predicts and returns the class of a document based on the prior probablities of class,likelihood of words.

    ...

    Parameters
    ----------
    test_document : Text
      testing sentence.
    smoothing : Text
      String representing the method to be followed for smoothing.
      refer page number 5-6 of https://web.stanford.edu/~jurafsky/slp3/4.pdf thi lesson to better understand smoothing
      smoothing should be laplace for laplace smoothing. 
    """
    x = y = 0
    if(smoothing == "laplace"):
      x = 1
      y = len(self.vocabulary) 
    posteriors = {}
    for cls in self.class_priors:
      likelihood = 0 
      for word in test_document:
        likelihood += np.log((self.classes_and_each_words_freq__dict[cls].get(
            word, 0)+x)/(self.classes_word_count_dict[cls]+y))
      posteriors[cls] = likelihood+self.class_priors[cls] 
    return max(posteriors,key = lambda x:posteriors[x])
