class Document:
    def __init__(self, id="", title:str="", abstract:str="", preprocessed_title:list[str]=[], preprocessed_abstract:list[str]=[], preprocessed_corpus:list[str]=[]):
        self.id = id
        self.title = title
        self.abstract = abstract
        self.preprocessed_title = preprocessed_title
        self.preprocessed_abstract = preprocessed_abstract
        self.preprocessed_corpus = preprocessed_corpus
        