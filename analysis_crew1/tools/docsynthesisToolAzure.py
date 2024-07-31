
    



import os
import json
import re
from typing import Dict, List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader,  PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from crewai_tools import tool
from langchain.agents import Tool
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain.vectorstores.azuresearch import AzureSearch
from panel_interface import chat_interface
from langchain_openai import AzureChatOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ComplexField,
    CorsOptions,
    SearchIndex,
    SearchFieldDataType,
    SimpleField,
    SearchableField
)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'))
embeddings = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002", model="text-embedding-ada-002", chunk_size=10)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
semantic_text_splitter = SemanticChunker(embeddings, breakpoint_threshold_amount="interquartile")
deployment_name3 = "gpt-35-turbo-16k"
deployment_name4 = "gpt-4"
llm_gpt3 = AzureChatOpenAI(deployment_name=deployment_name3, model_name=deployment_name3, temperature=0, streaming=True)
llm_gpt4 = AzureChatOpenAI(deployment_name=deployment_name4, model_name=deployment_name4, temperature=0, streaming=True)

endpoint = os.environ["SEARCH_ENDPOINT"]
key = os.environ["SEARCH_API_KEY"]
client = SearchIndexClient(endpoint, AzureKeyCredential(key))

sem_text_splitter = SemanticChunker(
   embeddings, breakpoint_threshold_type="interquartile"
)









class AzureDocumentSynthesisTool:
    @tool("Document_Synthesis")
    def azure_synthesize_documents(query: str, documents: List[Dict]):
        """
        Synthesizes single or multiple documents by extracting key information, themes, and narratives. 
        Provides a comprehensive and accessible overview of all relevant findings.
        Parameters:
            query (str): The user query to be answered.
            documents (list): A list of dictionaries containing details about the documents to be analyzed.
        Returns:
            str: A comprehensive synthesis of the information relevant to the query.
        """
        synthesized_information = ""
        for document in documents:
            
            title = document['title']  # Access the title of the document
            path = document['path']  # Document path
            file_name = os.path.basename(path)
            #print("Title of document is: ", file_name, end='\n')
            #print("path of doc is: ", path)
            # user_response = input(chat_interface.send("Would you like to use Azure for your vector index database?"
            #                                           "Any response other that Yes or Y (not case sensitive) will be treated as No.)", user="System"))
            # if user_response.lower() not in ("y", "yes"):
            #     chat_interface.send("We will use FAISS for vectorization and search", user="System")
            chat_interface.send(f"Preparing: {file_name} for processing", user="System")
            try:
                if title.endswith(".txt"):
                    loader = TextLoader(path)
                    chat_interface.send("Processing : ", file_name)  
                elif file_name.endswith('.pdf'):
                    chat_interface.send(f"Processing: {file_name} ", user = "Assistant") 
                    #loader = PyPDFLoader(full_file_path)
                    loader = PDFMinerLoader(path)
                elif title.endswith('.docx'):
                    loader = Docx2txtLoader(path)
                    chat_interface.send("Processing: ", file_name) 
                else:
                    print('The document format is not supported!') 
                     #loader = PyPDFLoader(full_file_path)
                document = loader.load()
                for page in document:
                    text="" 
                    text += page.page_content
                    #print("Preparing text for: ", title) 
                text = text.replace('\t', ' ')
                text = text.replace('\t', ' ')
                text= text.replace("\n", ' ')
                text = re.sub(" +", " ", text)
                text = re.sub("\u2022", "", text)
                text = re.sub(" +", " ", text)
                text = re.sub(r"\.{3,}", "", text)
                #print("This is the text: ", text, end='\n')
                    #chunks = text_splitter.create_documents([text])
                sem_chunksCD = sem_text_splitter.create_documents([text])
                #print("Prepared semcd chunks for: ", title) 
                    #loader = PyPDFLoader(path)
                    #needed_document = loader.load_and_split()
                    #needed_doc_chunks = text_splitter.split_documents(needed_document)

                #Provide choice via user input for in memeory FAISS or Azure AI Search index

                # Using FAISS for  similarity search
               
                index_name = "vector-index"
                azure_vector_store = AzureSearch(
                        azure_search_endpoint=os.environ.get("SEARCH_ENDPOINT"),
                        azure_search_key=os.environ.get("SEARCH_API_KEY"),
                        index_name=index_name,
                        embedding_function=embeddings.embed_query,
                    )
                azure_vector_store.add_documents(documents=sem_chunksCD)
                az_retriever = azure_vector_store.as_retriever()

               
                    # Running a retrieval-based QA
                qa_chain = RetrievalQA.from_chain_type(llm=llm_gpt4, retriever=az_retriever)
                    #qa_chain_mmr = RetrievalQA.from_chain_type(llm=llm_gpt4, retriever=retreiver_mmr)
                    # qa_chain_stuff = RetrievalQA.from_chain_type(llm=llm_gpt4,chain_type="stuff", retriever=retriever)
                    # qa_chain_srcdoc = RetrievalQA.from_chain_type(
                    # llm=llm_gpt4, chain_type="map_reduce", retriever=retriever, return_source_documents=True)
                    #result = qa_chain({"query": query})
                
                result = qa_chain.invoke({"query": query})
                synthesized_information += f"Title: {title}\n{result['result']}\n\n"
                name = index_name
                client.delete_index(name)   

            except Exception as e:
                print(f"An error occurred while processing document {title}: {e}")
        
        return synthesized_information



