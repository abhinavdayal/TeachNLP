import os
from typing import Text
DATA_EXTRACTION = "data_extraction" #Data folder
TEXT_FILES = "text_files" #Text files folder can be none if none represent it as  ""

    
class ReadData:
    """
    Read Data
    """
    def readData(self) -> Text:
        """
        method
        """
        text_file_names = os.listdir(os.path.join(DATA_EXTRACTION, TEXT_FILES))
        text_str = ""
        for file_name in text_file_names:
            with open(os.path.join(DATA_EXTRACTION, TEXT_FILES, file_name)) as f:
                text_str += f.read()
        return text_str


