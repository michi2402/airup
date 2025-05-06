import time

import requests
import logging
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def construct_query_from_key_terms(key_terms):
    if not key_terms or len(key_terms) == 0:
        return ""
    return ' AND '.join(key_terms.replace(' ', '+') for key_terms in key_terms)

def get_document_ids(query, ret_max, min_date, max_date):
    time.sleep(0.34)
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    dp_param = "db=pubmed"
    retmax_param = f"retmax={ret_max}"
    term_param = f"term={query}"
    date_range = f"mindate={min_date}&maxdate={max_date}"
    url = f"{base_url}?{dp_param}&{retmax_param}&{term_param}&{date_range}"

    response = requests.get(url)

    if response.status_code == 200:
        content = ET.fromstring(response.content)
        logger.info(f"Fetched {len(content.findall('.//IdList/Id'))} document IDs.")
        ids = [i.text for i in content.findall(".//IdList/Id")]
        return ids
    else:
        logger.error(f"Error fetching document IDs: {response.status_code}")
        return []

def get_documents(ids):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={','.join(ids)}&retmode=xml"

    response = requests.get(url)

    articles = {}
    if not response.status_code == 200:
        logger.error(f"Error fetching documents: {response.status_code}")
        return articles

    root = ET.fromstring(response.content)

    for article in root.findall(".//PubmedArticle"):
        pmid = article.find(".//PMID").text
        title = article.find(".//ArticleTitle").text
        abstract = article.find(".//AbstractText").text if article.find(".//AbstractText") is not None else ""
        articles["http://www.ncbi.nlm.nih.gov/pubmed/" + pmid] = {
            "title": title,
            "abstract": abstract
        }
    logger.info(f"Fetched {len(articles)} documents.")
    return articles