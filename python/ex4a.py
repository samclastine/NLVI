import warnings
import os
import json
import pandas as pd
from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.memory import ConversationBufferWindowMemory

from langchain.document_loaders import CSVLoader

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA

from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate



import time

from langchain.prompts import (
    PromptTemplate,

)
import re

import urllib.request
from langchain_openai import OpenAIEmbeddings

warnings.filterwarnings('ignore')
from models import initialize_evllm, initialize_openai_model, initialize_vllm, initialize_deepseek_model


class python_ex4a:
    def __init__(self, model_id, output_filename="/output.csv", mode="openai", JsonEnforcer= 'False'):
        self.model_id = model_id
        self.mode = mode
        self.output_filename = output_filename
        if self.mode == "hf" and JsonEnforcer == 'False':
            self.llm = initialize_vllm(model_id=self.model_id, temperature=0.3)
        if self.mode == "hf" and JsonEnforcer == 'True':
            self.llm = initialize_evllm(model_id=self.model_id, temperature=0.3)
        if self.mode == "openai":
            self.llm = initialize_openai_model(model_id=self.model_id, temperature=0.3)
        if self.mode == "deepseek":
            self.llm = initialize_deepseek_model(model_id=self.model_id, temperature=0.3)
        self.visualization_template = """/
                Generate a python code for the given question.\

                Note:
                1. load the csv file as pandas dataframe from the given directory "/content/data.csv". \
                2. save the chart in the following path "/content/drive/MyDrive/NLVI-Results/python/gpt4/charts/{question}.png". , use underscore inplace of space\
                3. Use matplotlib Library.


                previous conversation:
                {history}
                Data:
                {context}
                Question: {question}
                Python Code:"""





        self.memory = ConversationBufferWindowMemory(
            memory_key="history",
            k=1,
            input_key='question',
            return_messages=True
        )
        self.results = []
        self.data_url = None
        self.res = 'not'

    def visQA_chain(self, dataFile, input):
        if dataFile == "superstore":
            data_url = "https://raw.githubusercontent.com/nl4dv/nl4dv/master/examples/assets/data/" + dataFile + ".csv"
        else:
            data_url = "https://raw.githubusercontent.com/nlvcorpus/nlvcorpus.github.io/main/datasets/" + dataFile + ".csv"
        try:
            urllib.request.urlretrieve(self.data_url, 'data.csv')
            csv_loader = CSVLoader(file_path='/content/data.csv')
            csv_data = csv_loader.load()
            csv_text_splitter = CharacterTextSplitter(
                chunk_size=1000, chunk_overlap=0
            )
            csv_docs = csv_text_splitter.split_documents(csv_data)
            embeddings = OpenAIEmbeddings(model= "text-embedding-3-small")
            csv_retriever = FAISS.from_documents(csv_docs, embeddings).as_retriever()

            VIS_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question", "history"],partial_variables={"fileName":dataFile},template=self.visualization_template)

            vis_chain = RetrievalQA.from_chain_type(
                self.llm,
                retriever=csv_retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": VIS_CHAIN_PROMPT,"verbose":True,"memory": self.memory}
            )
            result = vis_chain({"query": input})
            data = result["result"]
            error='NAN'
            try:
                pattern = r'```python(.*?)```'
                match = re.search(pattern, data, re.DOTALL)
                code_block = match.group(1).strip() if match else ""
                print("exection started")
                exec(code_block)
                print("execution successful")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                error=e

            return {"data":data,"error":error}

        except(SyntaxError, ValueError) as e:
            print(f"Error in visQA chain func: {str(e)}")


    def append_result(self, result):
      # Define the set of required keys
      required_keys = {"datafile", "query", "actual", "predicted", "gpt_eval_score", "jcomp_score", "bleu1_score", "bleu2_score", "rouge1_score", "rouge2_score", "error"}
      
      # Fill in missing keys with default None values
      for key in required_keys:
          if key not in result:
              result[key] = None
      self.results.append(result)



    def generate(self, query,dataFile):
        try:
            predicted = self.visQA_chain(dataFile, query)
            eval_result = {
                    "datafile": dataFile,
                    "query": query,
                    "predicted": predicted["data"],
                    "error": predicted["error"]
                }
            self.append_result(eval_result)
            return predicted
        except Exception as e:
            errot=e
            print(f"An error occurred: {str(e)}")
            return "failed to generate"
  
    def write_to_csv(self):
            # Ensure there are results to write
            if not self.results:
                print("No results to write.")
                return

            # Create a DataFrame from results
            result_df = pd.DataFrame(self.results)
            print("DataFrame to be written:\n", result_df)  # Debugging line to see what is being written

            try:
                # Check if the file exists; append if yes, write new if no
                if os.path.isfile(self.output_filename):
                    result_df.to_csv(self.output_filename, mode='a', header=False, index=False)
                else:
                    result_df.to_csv(self.output_filename, index=False)
                print("Results successfully written to CSV.")
            except Exception as e:
                print(f"Failed to write to CSV: {str(e)}")  # Exception handling to capture and log any errors during the write operation

    def run_evaluation(self, queries_df):
        for index, row in queries_df.iterrows():
            # if index==1:
            #   break
            query = row['Utterance Set']
            Datafile = row['dataset'].lower()
            result = self.generate(query, Datafile)        
            if not result:
                    continue
            self.write_to_csv()
            return "Evaluation Process Completed!!!"