# Fill in missing values and rename config.example.py to config.py

# Entrez API Information
API_KEY = "" # Enter your API_KEY
E_UTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
E_SEARCH_URL = E_UTILS_BASE_URL + "esearch.fcgi"
E_FETCH_URL = E_UTILS_BASE_URL + "efetch.fcgi"
RETMAX = "10000"
SORT = "relevance"
RETMODE = "xml"
PUBMED_DOC_URL_PREFIX = "http://www.ncbi.nlm.nih.gov/pubmed/" # after / there is the id of the doc

# set one of these to True and the other to False
USE_STEMMING = False
USE_LEMMATIZATION = True

# Set these paths for word2vec model
WORD2VEC_MODEL_PATH = "/word2vec/word2vecTools" # Enter path where you downloaded word2vec model
WORD2VEC_MODEL_SAVE_PATH = "/word2vec/word2vecTools/model.kv" # Enter path where you will store kv


# Output path for results in json format
OUTPUT_DIRECTORY_PATH = "output"
OUTPUT_FILENAME = "results.json"


DROP_OFF_RATIO = 0.12 # Drop-off ratio 
RELEVANCE_THRESHOLD = 0.7 # how relevant must items be compared to the best ranked item

SNIPPET_MIN_SCORE = 10 # min threshold for the snippet to be considered relevant

# spacy model
spacy_model = "en_core_sci_sm" 
