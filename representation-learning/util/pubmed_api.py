import time

from Bio import Entrez
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PubMedAPI:
    def __init__(self, email: str, max_results: int = 100):
        """
        Initialize the PubMed API client.
        :param email: Email address for NCBI Entrez API
        """
        Entrez.email = email
        self.max_results = max_results

    def fetch_pubmed(self, query: str) -> dict:
        """
        Search PubMed for articles matching the query.
        :param query: the search query
        :return: a dictionary containing article informations
        """
        time.sleep(0.34)  # NCBI API rate limit (3 requests per second)
        ids = self.__search_articles(query)
        if not ids:
            logger.warning(f"No articles found for query: {query}")
            return {}

        results = self.__fetch_articles(ids)
        if not results:
            logger.warning(f"No abstracts found for query: {query}")
            return {}

        return results

    def __search_articles(self, query: str) -> list[str]:
        """
        Search PubMed for articles matching the query.
        :param query: the search query
        :return: a list of PubMed IDs (PMIDs)
        """
        logger.info(f"Searching PubMed for query: {query}")
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=self.max_results)
            record = Entrez.read(handle)
            id_list = record["IdList"]
            handle.close()
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []

        if not id_list:
            logger.warning(f"No results found for query: {query}")
            return []

        logger.info(f"Found {len(id_list)} articles for query")
        return id_list

    def __fetch_articles(self, pmids: list[str]) -> dict:
        """
        Fetch title and abstracts for a list of PubMed IDs.
        :param pmids: list of PubMed IDs
        :return: dictionary mapping PMIDs to abstracts
        """
        logger.info(f"Fetching abstracts for {len(pmids)} articles")
        try:
            handle = Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml")
            records = Entrez.read(handle)
            handle.close()
        except Exception as e:
            logger.error(f"Error fetching abstracts: {e}")
            return {}

        articles = {}
        for record in records["PubmedArticle"]:
            pmid = "http://www.ncbi.nlm.nih.gov/pubmed/" + record["MedlineCitation"]["PMID"]
            title = record["MedlineCitation"]["Article"]["ArticleTitle"]
            abstract = record["MedlineCitation"].get("Article", {}).get("Abstract", {}).get("AbstractText",
                                                                                            ["No abstract available"])[
                0]
            articles[pmid] = {"title": title, "abstract": abstract}

        logger.info(f"Fetched abstracts for {len(articles)} articles")
        for article in articles:
            t = articles[article]["title"]
            a = articles[article]["abstract"]
            logger.debug(f"PMID: {article}, Title: {t}, Abstract: {a}")
        return articles
