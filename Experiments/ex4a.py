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
from models import initialize_evllm



class VegaLiteEvaluator_EX4A:
    def __init__(self, model_id, output_filename="/output.csv"):
        self.model_id = model_id
        self.evaluator = GPTEvaluator()
        self.llm = initialize_evllm(model_id= self.model_id, temperature=0.3)
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
            embeddings = OpenAIEmbeddings()
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


    def generate(self, query, dataFile, truth):
        try:
            predicted = self.visQA_chain(dataFile,query)
            try:
                pred = predicted
                # pred = predicted.replace('true', 'True')
            except (SyntaxError, ValueError) as e:
                print("Invalid prediction", e)
                eval_result = {
                    "datafile": dataFile,
                    "query": query,
                    "predicted": pred,
                    "error": "Invalid prediction" + str(e)
                }
                self.results.append(eval_result)
                self.write_to_csv()  # Write the result to CSV
                return "Invalid prediction"

            # Print the JSON strings for debugging
            print("Predicted JSON:", pred)
            print("Truth JSON:", truth)


            try:
                truth_json = json.loads(truth)
                truth_json['data'].clear()
                truth_json['data']['url'] = self.data_url
                truth_str = json.dumps(truth_json)
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing JSON: {str(e)}")

            # Ensure 'pred' and 'truth' are valid JSON strings
            try:
                eval_result = None
                _error = None
                try:
                    pred_json = json.loads(pred)
                    pred_json['data'].clear()
                    pred_json['data']['url'] = self.data_url
                    truth_json = ast.literal_eval(truth)

                    pred_str = json.dumps(pred_json)
                    jcomp = JSONComparator(pred_json, truth_json)
                    jcomp_score = jcomp.evaluate_json()
                    bleu1_score = Bleu_1_score(pred, truth)
                    bleu1_score = bleu1_score.evaluate_bleu()
                    bleu2_score = bleu_2_score(pred, truth)
                    bleu2_score = bleu2_score.evaluate_bleu()
                    rouge1_score = rouge_1_score(pred_json, truth_json)
                    rouge1_score = rouge1_score.evaluate_rouge()
                    rouge2_score = rouge_2_score(pred_json, truth_json)
                    rouge2_score = rouge2_score.evaluate_rouge()

                    eval_response = self.evaluator.run(query, dataFile, pred_str)

                    # Access the content
                    content = eval_response.choices[0].message.content

                    # Check the type of the content and handle it accordingly
                    if isinstance(content, str):
                        try:
                            gptScore = ast.literal_eval(content)
                            print("Evaluated Score:", gptScore)
                            eval_result = {
                                "datafile": dataFile,
                                "query": query,
                                "predicted": pred_str,
                                "gpt_eval_score": gptScore['Score'],
                                "jcomp_score": jcomp_score,
                                "bleu1_score": bleu1_score,
                                "bleu2_score": bleu2_score,
                                "rouge1_score": rouge1_score,
                                "rouge2_score": rouge2_score,
                                "error": _error
                            }
                            self.results.append(eval_result)
                            self.write_to_csv()  # Write the result to CSV
                        except ValueError as e:
                            print(f"Error evaluating content: {str(e)}")
                    else:
                        # If content is not a string, handle the integer or other types as needed
                        print(f"Content is not a string, but a {type(content).__name__}: {content}")

                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing JSON: {str(e)}")
            except (SyntaxError, ValueError):
                print("Invalid JSON in 'pred'")
                return "failed to evaluate"
        except (SyntaxError, ValueError):
            print("Invalid JSON")
            return "failed load Json"

    def write_to_csv(self):
        result_df = pd.DataFrame(self.results)
        if os.path.isfile(self.output_filename):
            result_df.to_csv(self.output_filename, mode='a', header=False, index=False)
        else:
            result_df.to_csv(self.output_filename, index=False)

    def run_evaluation(self, queries_df):
        for index, row in queries_df.iterrows():
            query = row['Utterance Set']
            vlSpec_output = row['VegaLiteSpec']
            Datafile = row['dataset'].lower()
            self.generate(query, Datafile, vlSpec_output)
        return "Evaluation Process Completed!!!"
