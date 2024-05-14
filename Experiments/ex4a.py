import warnings
import os
import json
import pandas as pd
import ast

from langchain.prompts import PromptTemplate
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain.document_loaders import CSVLoader
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts import (
    PromptTemplate,
)
from Evaluator import Bleu_1_score, bleu_2_score, rouge_1_score, rouge_2_score, GPTEvaluator, JSONComparator
import urllib
warnings.filterwarnings('ignore')
from models import initialize_evllm, initialize_openai_model

class VegaLiteEvaluator_EX4A:
    def __init__(self, model_id, output_filename="/output.csv", mode="openai"):
        self.model_id = model_id
        self.mode = mode
        self.evaluator = GPTEvaluator()
        if self.mode == "hf":
            self.llm = initialize_evllm(model_id=self.model_id, temperature=0.3)
        elif self.mode == "openai":
            self.llm = initialize_openai_model(model_id=self.model_id, temperature=0.3)
        self.output_filename = output_filename
        self.visualization_template = """/
Generate Vegalite JSON Specification for given question. \n \n

Data: \n {context} \n

Current conversation: \n
{history} \n
Question: {question} \n
Vega-lite Json: """
        self.VIS_CHAIN_PROMPT = PromptTemplate(input_variables=["history", "question"], template=self.visualization_template)
        self.results = []
        self.data_url = None
        self.memory = ConversationBufferWindowMemory(
                memory_key="history",
                input_key="question",
                k=1
            )
    def visQA_chain(self, dataFile, input):
        if dataFile == "superstore":
            self.data_url = "https://raw.githubusercontent.com/nl4dv/nl4dv/master/examples/assets/data/" + dataFile + ".csv"
        else:
            self.data_url = "https://raw.githubusercontent.com/nlvcorpus/nlvcorpus.github.io/main/datasets/" + dataFile + ".csv"
        try:
            urllib.request.urlretrieve(self.data_url, dataFile)
            csv_loader = CSVLoader(file_path=dataFile)
            csv_data = csv_loader.load()
            csv_text_splitter = CharacterTextSplitter(
                chunk_size=1000, chunk_overlap=0
            )
            csv_docs = csv_text_splitter.split_documents(csv_data)
            embeddings = OpenAIEmbeddings(model= "text-embedding-3-small")
            csv_retriever = FAISS.from_documents(csv_docs, embeddings).as_retriever()


            
            vis_chain = RetrievalQA.from_chain_type(
                self.llm,
                retriever=csv_retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": self.VIS_CHAIN_PROMPT,"verbose":False,"memory": self.memory}
            )
            result = vis_chain({"query": input})
            result = result["result"]
            return result
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
    def generate(self, query, dataFile, truth):
        pred_str = None
        truth_str =  None
        try:
            predicted = self.visQA_chain(dataFile,query)
            pred = predicted

            try:
                truth = truth.replace('true', 'True')
                truth_json = ast.literal_eval(truth)
                truth_json['data'].clear()
                truth_json['data']['url'] = self.data_url
                truth_str = json.dumps(truth_json)
                truth_str = truth_str.replace('True', 'true')
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing JSON: {str(e)}")
                eval_result = {
                    "datafile": dataFile,
                    "query": query,
                    "actual": truth,
                    "predicted": pred,
                    "error": "Error parsing Truth JSON:" + str(e)
                }
                self.append_result(eval_result)
                return self.results

            # Ensure 'pred' and 'truth' are valid JSON strings
            try:
                eval_result = None
                _error = None
                try:
                    pred = pred.replace('true', 'True')
                    pred_json = json.loads(pred)
                    pred_json['data'].clear()
                    pred_json['data']['url'] = self.data_url
                    pred_str = json.dumps(pred_json)
                    pred_str = pred_str.replace('True', 'true')


                    jcomp = JSONComparator(pred_json, truth_json)
                    jcomp_score = jcomp.evaluate_json()
                    bleu1_score = Bleu_1_score(pred_str, truth_str)
                    bleu1_score = bleu1_score.evaluate_bleu()
                    bleu2_score = bleu_2_score(pred_str, truth_str)
                    bleu2_score = bleu2_score.evaluate_bleu()
                    rouge1_score = rouge_1_score(pred_json, truth_json)
                    rouge1_score = rouge1_score.evaluate_rouge()
                    rouge2_score = rouge_2_score(pred_json, truth_json)
                    rouge2_score = rouge2_score.evaluate_rouge()

                    # eval_response = self.evaluator.run(query, dataFile, pred_str)

                    # # Access the content
                    # content = eval_response.choices[0].message.content

                    # # Check the type of the content and handle it accordingly
                    # if isinstance(content, str):
                    try:
                        # gptScore = ast.literal_eval(content)
                        # if isinstance(gptScore, dict) and 'Score' in gptScore:
                        #     gpt_score = gptScore['Score']
                        # else:
                        #     gpt_score = None  # or some other error handling

                        # print("Evaluated Score:", gptScore)
                        eval_result = {
                            "datafile": dataFile,
                            "query": query,
                            "actual": truth_str,
                            "predicted": pred_str,
                            "gpt_eval_score": gpt_score,
                            "jcomp_score": jcomp_score,
                            "bleu1_score": bleu1_score,
                            "bleu2_score": bleu2_score,
                            "rouge1_score": rouge1_score,
                            "rouge2_score": rouge2_score,
                            "error": _error
                        }
                        self.append_result(eval_result)
                    except ValueError as e:
                            eval_result = {
                            "datafile": dataFile,
                            "query": query,
                            "actual": truth_str,
                            "predicted": pred_str,
                            "error": "Error evaluating content" + str(e)
                            }
                            self.append_result(eval_result)
                            print(f"Error evaluating content: {str(e)}")
                            return self.results
                    # else:
                    #     # If content is not a string, handle the integer or other types as needed
                    #     print(f"Content is not a string, but a {type(content).__name__}: {content}")

                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing JSON: {str(e)}")
                    eval_result = {
                            "datafile": dataFile,
                            "query": query,
                            "actual": truth,
                            "predicted": pred,
                            "error": "Error parsing JSON" + str(e)
                            }
                    self.append_result(eval_result)
                    print(f"Error parsing JSON: {str(e)}")
                    return self.results
            except (SyntaxError, ValueError):
                print("Invalid JSON in 'pred'")
        except (SyntaxError, ValueError):
            print("Invalid JSON")
            return False

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
            query = row['Utterance Set']
            vlSpec_output = row['VegaLiteSpec']
            Datafile = row['dataset'].lower()
            result = self.generate(query, Datafile, vlSpec_output)
            if not result:
                continue
        self.write_to_csv()
        return "Evaluation Process Completed!!!"
