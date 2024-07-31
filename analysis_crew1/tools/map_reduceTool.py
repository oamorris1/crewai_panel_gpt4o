import os
import json

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader,  PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.tools import tool
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import re
import tiktoken
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'))

embeddings = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002", model="text-embedding-ada-002", chunk_size=10)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len)
sem_text_splitter = SemanticChunker(
   embeddings, breakpoint_threshold_type="interquartile"
)
deployment_name4 = "gpt-4"

llm_gpt4 = AzureChatOpenAI(deployment_name=deployment_name4, model_name=deployment_name4, temperature=0, streaming=True)
#path = r'C:\Users\Admin\Desktop\erdcDBFunc\crewAIDocSum\documents'
path_summary =  r'C:\Users\Admin\Desktop\erdcDBFunc\analysis_crew\summaries'
#new_path = r'C:\Users\Admin\Desktop\erdcDBFunc\crewAIDocSum\new_documents'

def embedding_cost(chunks):
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in chunks])
    # print(f'Total tokens: {total_tokens}')
    # print(f'Cost in US Dollars: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens
 

map_template = """
           
           Throughly read, digest and anaylze the content of the documents. 
           Produce a thorough, comprehensive  summary that encapsulates the entire documents' main findings, methodology,
           results, and implications of the study. Ensure that the summary is written in a manner that is accessible to a general audience
           while retaining the core insights and nuances of the original paper. Include ALL key terms, definitions, descriptions, points of interest
            statements of facts and concepts, and provide any and all necessary context
           or background information. The summary should serve as a standalone piece that gives readers a comprehensive understanding
           of the documents' scope, significance, theme, meaning and conclusions without needing to read the entire document. Be as THOROUGH and DETAILED as possible.  You MUST
           include all concepts, techniques, variables, studies, research, main findings and conclusions.
           The summary MUST be long enough to capture ALL information in the document:
"{docs}"
Thorough SUMMARY:"""
map_prompt = PromptTemplate.from_template(map_template)

reduce_template = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary. Ensure that the final consolidated 
summary is written in a manner that is accessible to a general audience
while retaining the core insights and nuances of the original set of many summaries.
Include ALL key terms, definitions, descriptions, points of interest
statements of facts and concepts, and provide any and all necessary context
or background information. The final summary should serve as a standalone piece that gives readers a comprehensive understanding
of the documents' scope, significance, theme, meaning and conclusions without needing to read the entire set of summaries. Be as THOROUGH and DETAILED as possible.  You MUST
include all concepts, techniques, variables, studies, research, main findings and conclusions.
The final consolidated summary MUST be long enough to capture ALL information in the set of summaries: 
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)


class ObtainDocSummary():
   @tool("Document_Summary")
   def doc_sum(docs_path):
        """ Use this tool to access the document folder and summarize a document"""
        text="" 
        print("Begin loop")
        summaries =[]
        for file_Name in os.listdir(docs_path):
          full_file_path = os.path.join(docs_path, file_Name)
          #  code to handle other files than pdf  
          if file_Name.endswith(".txt"):
             loader = TextLoader(full_file_path) 
          elif file_Name.endswith('.pdf'):
            print("Preparing to sumarize: ", file_Name) 
            #loader = PyPDFLoader(full_file_path)
            loader = PDFMinerLoader(full_file_path)
          elif file_Name.endswith('.docx'):
            loader = Docx2txtLoader(file)
          else:
            print('The document format is not supported!') 
          #loader = PyPDFLoader(full_file_path)
          document = loader.load()
          for page in document:

            text += page.page_content
    
          text = text.replace('\t', ' ')
          text = text.replace('\t', ' ')
          text= text.replace("\n", ' ')
          text = re.sub(" +", " ", text)
          text = re.sub("\u2022", "", text)
          text = re.sub(" +", " ", text)
          text = re.sub(r"\.{3,}", "", text)
          chunks = text_splitter.create_documents([text])
          sem_chunksCD = sem_text_splitter.create_documents([text])
          
          num_tokens = embedding_cost(chunks)
          
            
            
          map_chain = LLMChain(llm=llm_gpt4, prompt=map_prompt)
          reduce_chain = LLMChain(llm=llm_gpt4, prompt=reduce_prompt)

          combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
        )
          reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
         collapse_documents_chain=combine_documents_chain,
     # The maximum number of tokens to group documents into.
         token_max=4000,

)
          map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)


        
          print("Preparing to run stuff chain")
          summary_dict = map_reduce_chain.invoke(sem_chunksCD)
          print("Chain complete") 
            
          summary = summary_dict.get("output_text")
          print("This is the summary: ", summary)
               
          new_file_name = file_Name.strip(".pdf")
          summaries.append({"title": file_Name, "summary":summary, "path":full_file_path})
          with open('summaries.json', 'w') as file:
               json.dump(summaries, file)  # Saving the list as JSON
          with open(f'{path_summary}\{new_file_name}_Summary.txt', "w") as file:
               file.writelines(summary)
          print("Done with summary write")
        return summaries