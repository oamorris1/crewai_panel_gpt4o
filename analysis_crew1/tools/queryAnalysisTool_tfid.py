import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os
import json
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.tools import tool
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import AzureChatOpenAI
import json
from langchain.tools import tool
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'))
import json
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#nltk.download('punkt')
#nltk.download('stopwords')
deployment_name4 = "gpt-4"

llm_gpt4 = AzureChatOpenAI(deployment_name=deployment_name4, model_name=deployment_name4, temperature=0, streaming=True)
path = r'C:\Users\Admin\Desktop\erdcDBFunc\crewAIDocSum\documents'
path_summary =  r'C:\Users\Admin\Desktop\erdcDBFunc\crewAIDocSum\summaries'

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

        # Initialize text preprocessing tools
        stemmer = PorterStemmer()
        english_stopwords = stopwords.words('english')

        def preprocess_text(text):
            tokens = word_tokenize(text.lower())
            return ' '.join([stemmer.stem(token) for token in tokens if token not in english_stopwords])

        # Prepare the query and summaries for TF-IDF vectorization
        documents = [preprocess_text(query)] + [preprocess_text(summary['summary']) for summary in document_summaries]
        vectorizer = TfidfVectorizer(stop_words='english', norm='l2')  # L2 normalization
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calculate cosine similarity between the query and each document summary
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Define a dynamic threshold based on query complexity
        def calculate_threshold(query):
            base_threshold = 0.1
            if len(query.split()) > 20:  # Adjust threshold for longer queries
                return base_threshold + 0.03
            return base_threshold

        threshold = calculate_threshold(query)

        # Filter out documents and keep scores separate
        relevant_documents = []
        irrelevant_documents= []
        document_scores = []

        for idx, score in enumerate(cosine_similarities):
            document = {
                "title": document_summaries[idx]['title'],
                "path": document_summaries[idx]['path'],
                "score": score
            }
            document_scores.append({
                "title": document_summaries[idx]['title'],
              
                "score": score
            })
            if score > threshold:
                #relevant_document = document_summaries[idx]
                relevant_documents.append(document)
                
            else:
                irrelevant_documents.append(document)
                
        # Return relevant documents and their scores separately
        return {"documents": relevant_documents, "irrelevant_documents": irrelevant_documents, "scores": document_scores}