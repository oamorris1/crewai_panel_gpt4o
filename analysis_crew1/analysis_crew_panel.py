from crewai import Agent, Task, Crew, Process
from crewai.agents import CrewAgentExecutor
from crewai.project import CrewBase, agent, crew, task
from langchain_core.callbacks import BaseCallbackHandler
from typing import TYPE_CHECKING, Any, Dict, Optional
from typing import Union, List, Tuple, Dict
from langchain_openai import AzureChatOpenAI
from langchain_core.agents import AgentFinish

from tools.queryAnalysisTool import QueryDocumentAnalysis
from tools.summaryTool import ObtainDocSummary
from tools.docsynthesisTool import DocumentSynthesisTool
#from tools.docsynthesisToolAzure import AzureDocumentSynthesisTool

import sys
import threading
import time 
from pathlib import Path
import json
from dotenv import load_dotenv, find_dotenv

from panel_interface import chat_interface, file_input, process_button, serve_app, register_start_crew_callback
from tasks import AnalyzeDocumentTasks


import panel as pn 
pn.extension(design="material")
uploaded_files = []
uploaded_filenames = []
load_dotenv(find_dotenv('.env'))

deployment_name3 = "gpt-35-turbo-16k"
deployment_name4 = "gpt-4"
llm_gpt3 = AzureChatOpenAI(deployment_name=deployment_name3, model_name=deployment_name3, temperature=0, streaming=True)
llm_gpt4 = AzureChatOpenAI(deployment_name=deployment_name4, model_name=deployment_name4, temperature=0, streaming=True)
deployment_name4o = "gpt-4o"
llm_gpt4o = AzureChatOpenAI(deployment_name=deployment_name4o, model_name=deployment_name4o, temperature=0, streaming=True)





def step_callback(
    agent_output: Union[str, List[Tuple[Dict, str]], AgentFinish],
    agent_name: str,
):
 
    if isinstance(agent_output, str):
        try:
            agent_output = json.loads(agent_output)
        except json.JSONDecodeError:
            pass

    if isinstance(agent_output, list) and all(
            isinstance(item, tuple) for item in agent_output
        ):

        for action, description in agent_output:
            # Send agent name
            chat_interface.send(f"Agent Name: {agent_name}", user="assistant", respond=False)
            # Send tool information
            chat_interface.send(f"The tool used: {getattr(action, 'tool', 'Unknown')}", user="assistant", respond=False)
            chat_interface.send(f"Tool input: {getattr(action, 'tool_input', 'Unknown')}", user="assistant", respond=False)
            chat_interface.send(f"{getattr(action, 'log', 'Unknown')}", user="assistant", respond=False)
            # Send observation in an expandable format , if observation is too verbose comment out
            #chat_interface.send(f"Observation: {description}", user="assistant", respond=False)

    elif isinstance(agent_output, AgentFinish):
        chat_interface.send(f"Agent Name: {agent_name}", user="assistant", respond=False)
        output = agent_output.return_values
        chat_interface.send(f"I have finished my task:\n{output['output']}", user="assistant", respond=False)

    else:
        chat_interface.send(f"Unexpected output type: {type(agent_output)}", user="assistant", respond=False)
        chat_interface.send(f"Output content: {agent_output}", user="assistant", respond=False)
class DocumentSummarizeAgents():
   


    def document_summary_agent(self):
        return Agent(
            role='Expert Research and Document Analyst',
            goal=f"""Obtain a document from the ObtainDocSummary tool.  
           Throughly read, digest and anaylze the content of the document. 
           Produce a thorough, comprehensive and clear summary that encapsulates the entire document's main findings, methodology,
           results, and implications of the study. Ensure that the summary is written in a manner that is accessible to a general audience
           while retaining the core insights and nuances of the original paper. Include key terms and concepts, and provide any necessary context
           or background information. The summary should serve as a standalone piece that gives readers a comprehensive understanding
           of the paper's significance without needing to read the entire document. Be as THOROUGH and DETAILED as possible.  You MUST
           include all concepts, techniques, variables, studies, research, main findings and conclusions. 
            """,
             backstory="""An expert writer, researcher and analyst. You are a renowned writer and researcher, known for
            your insightful and ability to write and summarize all key points in documents in an understable fashion.
            """,
    tools=[ObtainDocSummary.doc_sum],
    allow_delegation=False,
    verbose=True,
    max_iter=6,
    llm=llm_gpt4o,
    #callbacks=[MyCustomHandler("Summarizer")],
    step_callback = lambda output: step_callback(output, 'Summarizer')

        )

    def query_analysis_agent(self):
        return Agent(
            role="Expert Query Analyzer and Classifier",
            goal=f"""You receive user queries and determine the scope and depth of the required information to answer the query. Carefully analyze the query to extract
            what the user requires.
            Utilize the QueryAnalysisTool to dissect the query, identifying key words, phrases, and underlying questions.
            Classify the query to ascertain whether it can be addressed with a single document or if it requires a combination of documents.
            This classification should guide the subsequent agents in fetching and processing the right documents
            or summaries to formulate a complete and accurate response.""",
            backstory="""As a sophisticated linguistic model trained in semantic analysis and information retrieval, you specialize in understanding and categorizing complex queries.
            Your expertise lies in breaking down intricate questions into their elemental parts, determining the extent of information required,
            and directing these queries to the appropriate resources. Your analytical skills ensure that each query is processed efficiently and accurately, leading to timely and relevant responses.""",
            tools=[QueryDocumentAnalysis.analyze_query_and_summaries],
            allow_delgation=False,
            verbose=True,
            memory=True,
            llm=llm_gpt4,
            max_iter=6,
            step_callback = lambda output: step_callback(output, 'Query_Agent')
            
        )
        


        

    def document_analysis_agent(self):
      return Agent(
        role="Expert Integrative Synthesizer",
        goal=f""" Activated only after the query_analysis_agent has completed its assessment and identified the relevant documents necessary to address the user's query.
        Your primary function is to integrate and synthesize insights from multiple documents to formulate a comprehensive, nuanced response. 
        You conduct a deep examination into the content of each selected document, extracts vital themes, identifies discrepancies, and interconnect these
        findings to construct a detailed and insightful narrative that fully addresses the complexities of the query.
        The synthesis process is meticulous, aiming to provide a multifaceted answer that draws from a diverse array of sources,
        thereby enriching the final output with well-rounded perspectives.
        """,
        backstory="""As an advanced synthesis model equipped with cutting-edge NLP capabilities, you excel at integrating
        diverse pieces of information into a unified whole. Your skills enable you to discern patterns
        and connections between different data points, making you adept at handling complex queries that require insights from multiple perspectives.
        Your analytical prowess turns disparate documents into coherent narratives, making complex information accessible and understandable.""",
        tools=[DocumentSynthesisTool.synthesize_documents],
        allow_delegation=True,
        verbose=True,
        memory=True,
        llm=llm_gpt4,
        max_iter=6,
        step_callback = lambda output: step_callback(output, 'Analysis_Agent')
    )



docs_path = "C:/Users/omarmorris/Desktop/analysis_crew1/documents"
summaries_path = "C:/Users/omarmorris/Desktop/analysis_crew1/summaries.json"
agents = DocumentSummarizeAgents()
tasks = AnalyzeDocumentTasks()

def custom_ask_human_input(self, final_answer: dict) -> str:
      
      global user_input

      prompt = self._i18n.slice("getting_input").format(final_answer=final_answer)

      chat_interface.send(prompt, user="assistant", respond=False)

      while user_input == None:
          time.sleep(1)  

      human_comments = user_input
      user_input = None

      return human_comments


CrewAgentExecutor._ask_human_input = custom_ask_human_input

user_input = None
initiate_chat_task_created = False



query= "What are some common variables used in studies regarding human error-based aviation accidents "

#agents
summarizer_agent = agents.document_summary_agent()
analyzer_agent   = agents.query_analysis_agent()
docs_analyzer_agent = agents.document_analysis_agent()


def StartCrew(prompt):

    doc_sum_task = tasks.summarize_document(summarizer_agent, docs_path)
    analyze_query_task = tasks.analyze_document_query(analyzer_agent, summaries_path, prompt )
    docs_synthesizer_task = tasks.document_sythesis(docs_analyzer_agent, prompt)
   
    # Create the crew with a sequential process
    summary_crew= Crew(
    agents=[summarizer_agent, analyzer_agent,  docs_analyzer_agent],
    tasks=[doc_sum_task, analyze_query_task,  docs_synthesizer_task],
    process=Process.sequential,
    verbose=True,
    manager_llm=llm_gpt4
)


    results = summary_crew.kickoff()
    final_result = ""
    for idx, result in enumerate(results, 1):
        final_result += f"## Document {idx} Summary\n{result}\n\n"

    chat_interface.send("## Final Result\n"+result, user="assistant", respond=False)

register_start_crew_callback(StartCrew)
if __name__ == "__main__":
    serve_app()
