from typing import Any, Dict, List, Optional, Text
from .read_data import DATA_EXTRACTION, ReadData
import re
import os
import json
WORD_TOKENIZER_REGEX = r"(\W)"
SENTENCE_TOKENIZER_REGEX = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
WORD_TO_ID = "word_to_id.json"
ID_TO_WORD = "id_to_word.json"
SENTENCE_TOKENS = "sentence_tokens.txt"

class Tokenize:
    """
    Sentence tokenize and word tokenize and stores it in a list and Dict respectively
    """
    def __init__(self, data: Text, store_all: Optional[bool] = False) -> None:
        self.data = data
        self.word_to_id = {}
        self.id_to_word = {}
        self.store_all = store_all
    

    def getWordTokens(self) -> Dict:
        """
        Tokenizes each word and converts them into a Dictionary
        """
        self.temp_data = re.split(WORD_TOKENIZER_REGEX, self.data)
        self.data_as_set = set(self.temp_data)
        for index, word in enumerate(self.data_as_set):
            self.word_to_id[word] = index
            self.id_to_word[index] = word
        if self.store_all:
            with open(os.path.join(DATA_EXTRACTION, WORD_TO_ID), "w") as f:
                json.dump(self.word_to_id, f)
            with open(os.path.join(DATA_EXTRACTION, ID_TO_WORD), "w") as f:
                json.dump(self.id_to_word, f)
        return [self.word_to_id, self.id_to_word]
    

    def getSentenceTokens(self) -> List:
        """
        Tokenizes each sentence and converts them into a list
        """
        self.sentence_tokens = []
        self.sentence_tokens.append(re.split(SENTENCE_TOKENIZER_REGEX, self.data))
        if self.store_all:
            with open(os.path.join(DATA_EXTRACTION, SENTENCE_TOKENS), "w") as f:
                f.write(str(self.sentence_tokens))
        return self.sentence_tokens

if __name__ == "__main__":
    data = ReadData().readData()
    token_obj = Tokenize(data = data, store_all = True)
    token_obj.getWordTokens()
    token_obj.getSentenceTokens()



# print(Tokenize("sandeep is sandeep is sndeep. Mr. Sandeep is also sandeep").getSentenceTokens())