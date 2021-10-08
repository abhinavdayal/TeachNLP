import random
from typing import List, Optional, Text

class NGram:

    """
    * A class used used to represent n-gram language model for sentence generation.
    * The aim of am n-gram language model is to predict a word given previous n-1 words.
    * **Note** : tuple of n-1 words is referred as n-gram here. 
    * _For eg: ('he','is') is a 3-gram because it predicts 3rd word given previous two words._
    """


    def __init__(self, n: int, randomness_number: Optional[int] = 5) -> None:

        """
        Parameters
        ----------
        n : int
            A number representing the model. For example, n=2 will make a bi-gram model.

        randomness_number : int
            How many possible next words to consider for a given n-gram.

            * randomness_number = 1, will generate the same sentences again and again. 
            * A large randomness_number may include unrelated words.
            * The default value is 5

            ## Details
            For each n-gram, this will generate a key-value mapping every tuple of n-1 words to a list of length randomness_number,
            containing the most probable next words. i.e. `n-gram: [ w(1), w(2), .. , w(randomness_number) ]`.

            _For example_: For a 3-gram model with randomness_number=3, one possible key-value pair is: `(He,is):[a,good,nice]`.
        """
        self.n = n
        self.randomness_number = randomness_number

        self.__possible_next_words_dict = dict() # dict mapping n-gram to list of possible next words
        self.__vocabulary = set() # set of words in the training corpus

    def __n_gram_generator(self, sentence):
        """
        generates n-grams for the given sentence
        """
        window = ["<PAD>"]*(self.n-1)
        
        for token in sentence:
            self.__vocabulary.add(token)
            yield tuple(window), token
            window = window[1:] + [token]
        
        yield tuple(window), "<EOS>"


    def fit(self, training_corpus: List[List[Text]]) -> None:

        """
        fits the language model with  the given training  corpus.

        Parameters
        ----------
        training_corpus : List[List[Token]]
            It is a list of sentences. Each sentence is a list of tokens.
        
        Description
        -----------
        It uses [sliding window method](https://www.geeksforgeeks.org/window-sliding-technique) to scan eacn n-gram
        and first calculate the frequencies of each word following the n-gram. It then uses the frequencies to make 
        a list of top frequent (randomness_number) words for each n_gram.
        
        ## Example
        Let's see three iterations for a training sentence.
        
        **Actual sentence** : `['he','is','a','good','boy']`
        
        We will add <EOS> add to this sentence
        
        **sentence** : `['he','is','a','good','boy','<EOS>']`
    
        ```
        iteration1:
            window = ['<PAD>','<PAD>']
            frequency of the word 'he' is incremented by 1 in n_gram_freqs[('<PAD>','<PAD>')]
            window is updated with the word 'he'.
            window = ['<PAD>','he']

        iteration2:
            window = ['<PAD>','he']
            frequency of the word 'is' is incremented by 1 in n_gram_freqs[('<PAD>','he')]
            window is updated with the word 'is'.
            window = ['he','is']
        iteration3:
            window = ['he','is']
            frequency of the word 'is' is incremented by 1 in n_gram_freqs[('he','is')]
            window is updated with the word 'a'.
            window = ['is','a']
        ```
        """

        n_gram_freqs = {} # Temporary dictionary to store n-gram:{word:freq}
        # for a 3-gram model, at some instance of training it may look like {('<PAD>','<PAD>'):{'he':2,'A':3},(<'PAD>','he'):{'is':4,'eats:4}}
        window_size = self.n-1
        for tsentence in training_corpus:
            for n_gram, next_word in self.__n_gram_generator(tsentence):
                if n_gram not in n_gram_freqs:
                    n_gram_freqs[n_gram] = {}
                n_gram_freqs[n_gram][next_word] = n_gram_freqs[n_gram].get(next_word, 0)+1


        # populating self.__possible_next_words_dict with (n-gram:list of top randomness_number words) key-value pairs using n_gram_freqs
        for n_gram in n_gram_freqs:
            top_words = sorted(n_gram_freqs[n_gram], key=lambda x: n_gram_freqs[n_gram][x])
            self.__possible_next_words_dict[n_gram] = top_words[:self.randomness_number]


    def predict_next_word(self, seed: Optional[List[Text]] = []) -> Text:

        """
        Predict the next word for n-gram model, given the seed tokens

        Parameters
        ----------
        seed: List[Text]
          list of words to start with
        """

        #Pad with <PAD> token if seed is insufficient to forn a n-gram.
        if len(seed) < self.n-1:
            seed = ["<PAD>"]*(self.n-1-len(seed)) + seed

        # find the latest n-gram by picking the last n-1 words
        ngram = tuple(seed[-self.n+1:])
        return random.choice(self.__possible_next_words_dict.get(ngram, ['<EOS>']))


    def generate_sentence(self, seed: Text) -> Text:

        """
        Generates a sentence from a given seed.

        Parameters
        ----------
        seed: Text
          Words to start with
        """
        sentence = seed.split()
        next_word = self.predict_next_word(sentence)
        while next_word != '<EOS>':
            sentence += [next_word]
            next_word = self.predict_next_word(sentence)

        return " ".join(sentence)


