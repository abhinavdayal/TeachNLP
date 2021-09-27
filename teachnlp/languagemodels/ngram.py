import numpy 
import random 
from typing import Any, List, Text


class N_gram:
    def __init__(self, n_gram=2, randomness=20):
        """
        self.n_gram_dict = { (n-1)_gram:[top few words] }
        """
        self.n_gram = n_gram
        self.randomness = randomness 
        self.n_gram_dict = {}
        
        
    
    def fit(self,training_corpus):
        """
            fits the language model with  the given training  corpus  
            training corpus is list of sentences, each sentense is a list of words.
            [[s1],[s2],[s3],...]  <==> [[w11,w12,w13,..],[w21,w22,w23,...],[w31,w32,w33,...]]

            Sliding window method is used for n-gram count.
        """

        n_gram_dict = {}
        window_size = self.n_gram-1
        for sentence in training_corpus:
            window = ["<PAD>"]*window_size
            ptr2 = 0
            full_sentence = sentence[:]
            full_sentence.append("<EOS>")
            while(ptr2<len(full_sentence)):
                nth_words_freqt_dict = n_gram_dict.get(tuple(window),{}) 
                nth_words_freqt_dict[full_sentence[ptr2]] = nth_words_freqt_dict.get(full_sentence[ptr2],0)+1 
                n_gram_dict[tuple(window)] = nth_words_freqt_dict 
                window.pop(0)
                window.append(full_sentence[ptr2])
                ptr2 += 1 
        
        for n_gram in n_gram_dict: 
            top_words = sorted(n_gram_dict[n_gram],key = lambda x:n_gram_dict[n_gram][x])
            self.n_gram_dict[n_gram] = top_words[:self.randomness]
        print("Done!!!")
    

    def pad_list(self,list_of_words):
        """
            Padding with <PAD> tokens is done
        """
        padding = ["<PAD>"]*(self.n_gram-1-len(list_of_words))
        list_of_words = list_of_words[::-1]
        list_of_words.extend(padding)
        list_of_words = list_of_words[::-1]

    def strip_to_n(self,list_of_words):
        """
            stripping the list to n-1 words
        """
        list_of_words = list_of_words[::-1][:self.n_gram-1][::-1]

    def predict_the_next_word(self,beginning_words=[]):
        """
            this model takes previous n-1 words and predicts the next word for n-gram model.
            If length of the given words list is not equal to n-1 then Padding with <PAD> token is done.
        """
        self.strip_to_n(beginning_words)
        self.pad_list(beginning_words)
        if(tuple(beginning_words) in self.n_gram_dict):
            return "<EOS>"
        return random.choice(self.n_gram_dict[tuple(beginning_words)])

    def generate_sentence(self,beginning_words=[]):
        """
            generates a sentence from a given few beginning words.
        """
        sentence = beginning_words[:]
        window = beginning_words[:]
        self.strip_to_n(window)
        self.pad_list(window)
        next_word = self.predict_the_next_word(window) 
        while(next_word!="<EOS>"):
            sentence.append(next_word)
            window.pop(0)
            window.append(next_word)
            next_word = self.predict_the_next_word(window) 
        return " ".join(sentence)





            



    
