import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
import os
import json
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.tools import tool
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import json
from langchain.tools import tool
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'))
import json
#import ospi 
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

deployment_name4 = "gpt-4"
llm_gpt4 = AzureChatOpenAI(deployment_name=deployment_name4, model_name=deployment_name4, temperature=0, streaming=True)
deployment_name4o = "gpt-4o"
llm_gpt4o = AzureChatOpenAI(deployment_name=deployment_name4o, model_name=deployment_name4o, temperature=0, streaming=True)
path = r'C:\Users\omarmorris\Desktop\analysis_crew1\documents'
path_summary =  r'C:\Users\omarmorris\Desktop\analysis_crew1\summaries'

class QueryDocumentAnalysis:
    @tool("Query_and_Document_Summary_Analysis")
    def analyze_query_and_summaries(query, summaries_path):
        """
        Analyzes user queries against document summaries to determine relevant documents.
        Returns a list of dictionaries with the title and path of each relevant document and keeps the similarity scores separate.
        """
        try:
            with open(summaries_path, 'r') as file:
                document_summaries = json.load(file)
        except Exception as e:
            return {"error": str(e)}

        # Initialize OpenAI embeddings
        embeddings_model = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002", model="text-embedding-ada-002")
        

        # Embed the query
        query_embedding = embeddings_model.embed_query(query)

        # Embed the document summaries
        document_embeddings = []
        for summary in document_summaries:
            document_embedding = embeddings_model.embed_query(summary['summary'])
            document_embeddings.append({
                "title": summary['title'],
                "path": summary['path'],
                "embedding": document_embedding
            })

        # Calculate cosine similarity between the query embedding and each document summary embedding
        def cosine_similarity(vec1, vec2):
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        cosine_similarities = [
            cosine_similarity(query_embedding, doc['embedding']) for doc in document_embeddings
        ]

        # Define a dynamic threshold based on query complexity
        def calculate_threshold(query):
            base_threshold = 0.76
            if len(query.split()) > 20:  # Adjust threshold for longer queries
                return base_threshold + 0.03
            return base_threshold

        threshold = calculate_threshold(query)

        # Filter out documents and keep scores separate
        relevant_documents = []
        irrelevant_documents = []
        document_scores = []

        for idx, score in enumerate(cosine_similarities):
            document = {
                "title": document_embeddings[idx]['title'],
                "path": document_embeddings[idx]['path'],
                "score": score
            }
            document_scores.append({
                "title": document_embeddings[idx]['title'],
                "score": score
            })
            if score > threshold:
                relevant_documents.append(document)
            else:
                irrelevant_documents.append(document)

        # Return relevant documents and their scores separately
        return {"documents": relevant_documents, "irrelevant_documents": irrelevant_documents, "scores": document_scores}