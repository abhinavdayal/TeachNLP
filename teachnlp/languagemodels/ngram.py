import random
from typing import List, Optional, Text

class N_gram:

    """A class used used to represent n-gram language model for sentence generation.
    The aim of am n-gram language model is to predict a word given previous n-1 words.
    Note : tuple of n-1 words is referred as n-gram here. 
    For eg: ('he','is') is a 3-gram because it predicts 3rd word given previous two words.
    ...

    Attributes
    ----------
    n : int
        A number reprsenting the model. 2 for bi-gram model and n for n-gram model.

    randomness_number : int
        It represents the randomness of our model. If it is too high it may generate meaningless sentences.
        If it is very less then it may generate the same sentences again and again. Read possible_next_words_dict to better understand the use of this.

    possible_next_words_dict : {n-gram:[w1,w2,..,w(randomness_number)]}
        It is a dictionary. It contains every tuple of n-1 words as keys and list of most probable next words as values.
        For eg: (He,is):[a,good,nice] -> This can be a key-value pair for a 3-gram model with randomness_number = 3.

    vocabulary : set()
        Set of all words appear in training corpus.


    Methods
    -------
    fit(training_corpus: List[List[Text]])
        fits the n-gram model with the training corpus.

    pad_sentence(sentence: List[Text])
        Adds padding to the sentence with <PAD> tokens such that the length of the sentence becomes n-1 words in case of n-gram model
        (2 words in the case of Trigram model)

    strip_sentence(sentence: List[Text])
        Strips the extra words such that the length of the sentence becomes n-1 words in case of n-gram model(2 words in the case of Trigram model)

    predict_next_word(beginning_words: Optional[List[Text]])
        Takes few words and generates the next word. However it uses  n-1 words in case of n-gram model (2 words in the case of Trigram model)

    generate_sentence(beginning_words: Optional[List[Text]])
        Takes few words and generates a sentence.
    """


    def __init__(self, n: int, randomness_number: Optional[int] = 5) -> None:

        """
        Parameters
        ----------
        n : int
            A number reprsenting the model. 2 for bi-gram model and n for n-gram model.
        randomness_number : Optional[int]
            It represents the randomness of our model. If it is too high it may generate meaningless sentence.
            If it is very less then it may generate same sentences again and again
        """
        self.n = n
        self.randomness_number = randomness_number
        self.possible_next_words_dict = dict()
        self.vocabulary = set()



    def fit(self, training_corpus: List[List[Text]]) -> None:

        """fits the language model with  the given training  corpus.

        Parameters
        ----------
        training_corpus : List[List[Text]]
            It is a list of sentences. Each sentence is a list of words.
        
        Description
        -----------
        It uses sliding window method for n-gram count. You can read about sliding window here https://www.geeksforgeeks.org/window-sliding-technique 
        This method scans the entire training corpus. 
        For an n-gram model. It takes window of size n-1. Window slides from left to right.

        First we store (n-gram,{word:frequency}) in temp_n_gram_dict by scanning the entire training corpus.
        Then self.possible_next_words_dict is populates with (n-gram:list of top probable words) key-value pairs using temp_n_gram_dict.

        Sample Iterations
        -----------------
        For eg: Let's see three iterations for a training sentence.
        Actual sentence : ['he','is','a','good','boy']
        We will add <EOS> add to this sentence
        sentence : ['he','is','a','good','boy','<EOS>']
        iteration1:
            window = ['<PAD>','<PAD>']
            frequency of the word 'he' is incremented by 1 in temp_n_gram_dict[('<PAD>','<PAD>')]
            window is updated with the word 'he'.
            window = ['<PAD>','he']
        iteration2:
            window = ['<PAD>','he']
            frequency of the word 'is' is incremented by 1 in temp_n_gram_dict[('<PAD>','he')]
            window is updated with the word 'is'.
            window = ['he','is']
        iteration3:
            window = ['he','is']
            frequency of the word 'is' is incremented by 1 in temp_n_gram_dict[('he','is')]
            window is updated with the word 'a'.
            window = ['is','a']
        """

        temp_n_gram_dict = {} # Temporary dictionary to store n-gram:{word:freq}
        # for a 3-gram model, at some instance of training it may look like {('<PAD>','<PAD>'):{'he':2,'A':3},(<'PAD>','he'):{'is':4,'eats:4}}
        window_size = self.n-1
        for tsentence in training_corpus:
            window = ["<PAD>"]*window_size
            # Duplcating the sentence because we need to add <EOS> tag and sentences in the training corpus shouldn't be modified.
            sentence = tsentence[:]
            sentence.append("<EOS>")

            # loop to slide the window form left to right of the training sentence
            for ptr in range(len(sentence)):
                self.vocabulary.add(window[ptr])
                words_freqt_dict = temp_n_gram_dict.get(tuple(window), dict())

                # frequency of current word should be incremented by 1 in temp_n_gram_dict[tiple(Window)]
                words_freqt_dict[sentence[ptr]] = words_freqt_dict.get(sentence[ptr], 0)+1
                temp_n_gram_dict[tuple(window)] = words_freqt_dict
                window.pop(0)
                window.append(sentence[ptr])

        # populating self.possible_next_words_dict with (n-gram:list of top randomness_number words) key-value pairs using temp_n_gram_dict
        for n_gram in temp_n_gram_dict:
            top_words = sorted(temp_n_gram_dict[n_gram], key=lambda x: temp_n_gram_dict[n_gram][x])
            self.possible_next_words_dict[n_gram] = top_words[:self.randomness_number]
        print("Done!!!")



    def pad_sentence(self, sentence: List[Text]) -> List[Text]:

        """Padding with <PAD> tokens is done such that length of sentence becomes self.n - 1 
        For eg: If we trained a 3-gram model The ['good'] will become ['<PAD>','good'] after padding
        ...

        Parameters
        ----------
        sentence: List[Text]
          list of words representing the sentence
        """
        padding = ["<PAD>"]*(self.n-1-len(sentence))
        sentence = sentence[::-1]
        sentence.extend(padding)
        sentence = sentence[::-1]
        return sentence



    def strip_sentence(self, sentence: List[Text]) -> List[Text]:

        """Strips the sentence Such that length of sentence becomes self.n - 1 
        For eg: If we trained a 3-gram model The ['he','is','a','good'] will become ['a','good'] after stripping
        ...

        Parameters
        ----------
        sentence: List[Text]
          list of words representing the sentence
        """
        return sentence[::-1][:self.n-1][::-1]


    def process_sentence(self,sentence: List[Text]) -> List[Text]:
        """This method applies stripping followed by padding
        ...

        Parameters
        ----------
        sentence: List[Text]
          list of words representing the sentence
        """
        sentence = self.strip_sentence(sentence)
        sentence = self.pad_sentence(sentence)
        return sentence



    def predict_next_word(self, beginning_words: Optional[List[Text]] = []) -> Text:

        """This method takes few words and predicts the next word for n-gram model.
        If length of the given words list is less than self.n-1 then Padding with <PAD> token is done.
        If it beginning_words contains more than n-1 words then excess words are stripped.
        
        ...

        Parameters
        ----------
        beginning_words: List[Text]
          list of words
        """
        beginning_words = self.process_sentence(beginning_words)
        if tuple(beginning_words) not in self.possible_next_words_dict:
            return "<EOS>"
        return random.choice(self.possible_next_words_dict[tuple(beginning_words)])



    def generate_sentence(self, beginning_words: Optional[List[Text]] = []) -> Text:

        """Generates a sentence from a given few beginning words.
        Words are keep on generating till <EOS> is generated.
        ...

        Parameters
        ----------
        beginning_words: List[Text]
          list of words
        """
        sentence = beginning_words[:]
        window = beginning_words[:]
        window = self.process_sentence(window)
        next_word = self.predict_next_word(window)
        while (next_word != "<EOS>"):
            sentence.append(next_word)
            window.pop(0)
            window.append(next_word)
            next_word = self.predict_next_word(window)
        return " ".join(sentence)


