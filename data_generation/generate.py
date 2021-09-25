import re
import numpy as np
from typing import Optional, Text, List, Dict, Any
TOKEN_REGEX = r"[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*"

class PreProcessData:

    def __init__(self, text: Text, window_size: int) -> None:
        """
        To preprocess the text data into tokens and map them
        """
        self.text = text
        self.window_size = window_size
    
    def tokenize(self) -> List[Text]:
        """
        Tokenizes each document
        """
        pattern = re.compile(TOKEN_REGEX)
        self.tokens = pattern.findall(self.text) 
        return self.tokens

    def mapping(self) -> List:
        """
        Maps the word to id and id to word
        Returns:
            List of Dictionaries with word_to_id and id_to_word
        """
        self.word_to_id = {}
        self.id_to_word = {}

        for index, token in enumerate(set(self)):
            self.word_to_id[token] = index
            self.id_to_word[index] = token

        return [self.word_to_id, self.id_to_word]
    
    def generate_training_data(self) -> List:
        """
        Generats the training data in np arrays
        """
        N = len(self.tokens)
        self.X, self.Y = [], []
        for i in range(N):
            nbr_inds = list(range(max(0, i - self.window_size), i)) + list(range(i + 1, min(N, i + self.window_size + 1)))
            for j in nbr_inds:
                self.X.append(self.word_to_id[self.tokens[i]]) #For Center word
                self.Y.append(self.word_to_id[self.tokens[j]]) #For remaining context words
        
        self.X, self.Y = np.expand_dims(np.array(self.X)), np.expand_dims(np.array(self.Y))
        
        return self.X, self.Y



class Initialization:

    def __init__(self, vocab_size: Optional[int], embed_size: Optional[int], input_size: Optional[int], output_size: Optional[int]) -> None:
        """
        Initializes the Model parameters
        Args:
            vocab_size: int: The size of the vocablary
            embed_size: int: the size of the embeddings
            input_size: int: Input size of the dense layer
            output_size: int: Ouput size of the dense layer
        """
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.input_size = input_size
        self.output_size = output_size

    def __initializeDenseLayer(self) -> None:
        """
        Initializes the values required for Dense Layer
        Returns:
            A numpy array with size
        """
        W_DENSE_LAYER = np.random.randn(self.output_size, self.input_size) * 0.01
        return W_DENSE_LAYER

    def __initializeWordEmbeddings(self) -> None:
        """
        Intializes Values for Word Embeddings
        Returns:
            A numpy array
        """
        WRD_EMB = np.random.randn(self.vocab_size, self.embed_size) * 0.01
        return WRD_EMB

    def initializeParaMaters(self) -> Dict:
        """
        Initializes all the parameters required for initializing the models
        """
        WRD_EMB = self.__initializeWordEmbeddings()
        W_DENSE_LAYER = self.__initializeDenseLayer()
        parameters = {}
        parameters['WRD_EMB'] = WRD_EMB
        parameters['W_DENSE_LAYER'] = W_DENSE_LAYER
        return parameters


class ForwardPropagation:

    def __init__(self, inds: Optional[List], parameters: Optional[Dict], word_vec: Optional[List]) -> None:
        self.inds = inds
        self.parameters = parameters
        self.word_vec = word_vec

    def __ind_to_word_vecs(self):
        """
        Convers the batch array into word vectors
        """
        m = self.inds.shape[1]
        WRD_EMB = self.parameters.get("WRD_EMB")
        word_vec = WRD_EMB[self.inds.flatten(), :].T
        assert (word_vec.shape == (WRD_EMB.shape[1], m))

        return word_vec

    def __linear_dense(self):
        """
        Converts np arrays for Dense layer
        """
        m = self.word_vec.shape[1]
        W_DENSE_LAYER = self.parameters.get("W_DENSE_LAYER")
        self.Z = np.dot(W_DENSE_LAYER, self.word_vec)

        assert (self.Z.shape == (W_DENSE_LAYER.shape[0], m))

        return W_DENSE_LAYER, self.Z

    def __softmax(self):
        """
        Z: output out of the dense layer, shaper (vocab_size, m)
        """
        softmax_output = np.divide(np.exp(self.Z), np.sum(np.exp(self.Z), axis = 0, keepdims = True) + 0.001)
        assert (softmax_output == self.Z.shape)
        
        return softmax_output
    
    def forwardPropagation(self):
        word_vec = self.__ind_to_word_vecs(self.inds, self.parameters)
        W_LINEAR_DENSE, Z = self.__linear_dense(self.word_vec, self.parameters)
        softmax_out = self.__softmax()
        caches = {}
        caches['inds'] = self.inds
        caches['word_vec'] = word_vec
        caches['W_LINEAR_DENSE'] = W_LINEAR_DENSE
        caches['Z'] = Z

        return softmax_out, caches

class CostFunction:

    def __init__(self, softmax_out, Y) -> None:
        """
        Softmax output
        """
        self.softmax_out = softmax_out
        self.Y = Y
        m = self.softmax_out.shape[1]
        cost = -(1/m) * np.sum(np.sum(self.Y * np.log(self.softmax_out + 0.001), axis = 0, keepdims = True), axis = 1)
        return cost


class BackPropagation:
    pass


class Train:
    pass


class Metrics:
    pass



class Main:

    def __init__(self, doc: Text) -> None:
        pass
    
    def wordTwoVec(self):

        return

    pass