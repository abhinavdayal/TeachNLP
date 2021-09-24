from bs4 import BeautifulSoup
from  urllib.request import Request, urlopen 
from typing import Any, NoReturn, Optional, Text, Dict, List
import re
import os
url = "https://www.freechildrenstories.com"
hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}
DIV = "div"
A = "a"
HREF = "href"
SECTION = "section"
CLASS = "class"
TEXT_FOLDER = "text_files"
MIDDLE_PAGE_CSS =  "col sqs-col-4 span-4"
ALTERNATE_MIDDLE_PAGE_CSS = "col sqs-col-6 span-6"
CONTENT_PAGE_CSS = "Main-content"
MAIN_PAGE_CSS = re.compile(r'^/.*-.*$')
https_pattern = re.compile(r"^https://.*$")
count = 0


class Utilities:

    #Deprecated
    @classmethod
    def getKey(cls, dict_of_stories: Dict, to_find: Text) -> Text:
        for key in dict_of_stories.keys():
            if to_find in dict_of_stories[key]:
                return key[1:]

class ExtractDataFromURL:
    """
    Extract data from the given URL
    """

    def __init__(self, url) -> None:
        self.url = url
        self.request =  Request(self.url, headers = hdr)
        page = urlopen(self.request).read()
        self.Beautified = BeautifulSoup(page, 'lxml')
    

    def getBeautified(self) -> BeautifulSoup:
        return self.Beautified



class AnalyzeData:

    def __init__(self, find_tag: Text, find_dict: Dict, beautified_soap: BeautifulSoup, name_of_file: Optional[Text] = None, sub_folder_name: Optional[Text] = "") -> None:
        """
        Analyze the data and send the text in the data
        Args:
            page_type: Main page or content page
            find_tag: To find 'a' or 'section' tag
            find_dict: Finding dictionary like 'class' name or 'section' name
        Returns None
        """
        self.find_tag = find_tag
        self.find_dict = find_dict
        self.beautified_soap = beautified_soap
        self.links = []
        self.file_name = name_of_file
        self.subfolder_name = sub_folder_name

    def getLinksFromMainPage(self) -> List:
        """
        Gets the links from the main page
        Returns:
            The list of links to inside page
        """
        links = []
        links_dict = {}
        hrefs = self.beautified_soap.find_all(self.find_tag, self.find_dict)
        for index, value in enumerate(hrefs, 0):
            temp_link = hrefs[index]['href']
            temp_text = hrefs[index].text
            if temp_link not  in links:
                links.append(temp_link)
                links_dict[temp_link] = temp_text
        return [links, links_dict]

    #Deprecated
    def getDataFromContentPages(self) -> List[Any]:
        """
        Gets the text from the content page
        Returns:
            Returns the List of files
        """
        hrefs = self.beautified_soap.find_all(self.find_tag, self.find_dict)
        text = []
        for index, value in enumerate(hrefs, 0):
            temp = hrefs[index].text
            if len(temp) != 0:
                text.append(temp)
        return text
    
    def getLinksFromInsidePage(self) -> List[Any]:
        """
        Gets the links from the inside page
        Returns:
            List of links from the Inside page
        """
        hrefs = self.beautified_soap.find_all(self.find_tag, self.find_dict)
        links = []
        for row_data in hrefs:
            for column_data in row_data:
                try:
                    links.append(column_data.a['href'])
                except Exception as e:
                    continue             
        return links
    

    def getContentFromPage(self) -> None:
        """
        Gets the content from the Contents page and writes them in the text_files directory
        Creates files with the story name and count
        """
        global count #For global
        text = []
        content_raw = self.beautified_soap.find_all(self.find_tag, self.find_dict)
        for index, value in enumerate(content_raw, 0):
            try:
                temp = content_raw[index].text
                if len(temp) != 0:
                    text.append(temp)
                with open(os.path.join(TEXT_FOLDER, self.subfolder_name, f"{self.file_name}_{count}.txt"), "w+") as f:
                    count += 1
                    f.write("".join(text))
            except Exception as e: #Directory NOT found error
                try:
                    os.mkdir(os.path.join(TEXT_FOLDER, self.subfolder_name))
                except Exception as e: # File not found error
                    os.mkdir(TEXT_FOLDER)
                    os.mkdir(os.path.join(TEXT_FOLDER, self.subfolder_name))
                with open(os.path.join(TEXT_FOLDER, self.subfolder_name, f"{self.file_name}_{count}.txt"), "w+") as f:
                    count += 1
                    f.write("".join(text))
                continue
        return

class Main:

    def __init__(self) -> None:
        return
    
    def extractData(self) -> NoReturn:
        """
        Extract the data from the website
        """
        main_page_links, main_page_dict = AnalyzeData(A, {HREF : MAIN_PAGE_CSS}, ExtractDataFromURL(url = url).getBeautified()).getLinksFromMainPage() #Gets the page links and their dictionary
        inside_page_links = []
        inside_page_dict = {}
        for individual_links in main_page_links:
            temp_links = AnalyzeData(DIV, {CLASS : MIDDLE_PAGE_CSS}, ExtractDataFromURL(url = url +  individual_links).getBeautified()).getLinksFromInsidePage()
            if temp_links == []:
                temp_links = AnalyzeData(DIV, {CLASS : ALTERNATE_MIDDLE_PAGE_CSS}, ExtractDataFromURL(url = url +  individual_links).getBeautified()).getLinksFromInsidePage()
            inside_page_links.append(temp_links)
            inside_page_dict.update({individual_links : temp_links})
        final_story_links = []
        for content_page_link in inside_page_links:
            if content_page_link is not None or content_page_link != []:
                for individual_story_links in content_page_link:
                    if not re.match(https_pattern, individual_story_links):
                        final_story_links.append(individual_story_links)
        for story_link in final_story_links:
            sub_folder = Utilities.getKey(inside_page_dict, story_link) #Deprecated
            AnalyzeData(SECTION, {CLASS : CONTENT_PAGE_CSS}, ExtractDataFromURL(url = url + story_link).getBeautified(),  story_link[1:]).getContentFromPage()
if __name__ == "__main__":
    Main().extractData()