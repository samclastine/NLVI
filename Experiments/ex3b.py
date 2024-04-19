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
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from models import initialize_evllm

warnings.filterwarnings('ignore')



class VegaLiteEvaluator_EX3B:
    def __init__(self, model_id, output_filename="/output.csv"):
        self.model_id = model_id
        self.evaluator = GPTEvaluator()
        self.output_filename = output_filename
        self.results = []
        self.shots =  [
            {"question": "Scatter Plot for Relationship between Sales and Profit","output": """Let's think step by step:\nFirst, we need to specify the Vega-Lite schema: {{"$schema": "https://vega.github.io/schema/vega-lite/v4.json"}}\nNext, we should add a description:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json",\n"description": "Relationship between Sales and Profit"}}\n Now we need to define the data source URL:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json",\n"description": "Relationship between Sales and Profit",\n "data": {{"url": "https://example.com/data/sales-profit.json"}}}}\n For the plot, we want a scatter plot, so the mark is "point":\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json",\n"description": "Relationship between Sales and Profit",\n"data": {{"url": "https://example.com/data/sales-profit.json"}},\n"mark": "point"}}\nNext, we need to specify the encodings for x and y\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description": "Relationship between Sales and Profit","data": {{"url":"https://example.com/data/sales-profit.json"}},  "mark": "point","encoding": {{"x": {{...}},"y": {{...}}}}}}\nAnd define the fields, types, and axis titles:\nHere is the final Output:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "Relationship between Sales and Profit",  "data": {{"url": "https://example.com/data/sales-profit.json"}},"mark": "point","encoding": {{"x": {{"field": "sales", "type": "quantitative","axis": {{"title": "Sales"}},"y": {{"field": "profit","type": "quantitative","axis": {{"title": "Profit"}}}}"""},
            {"question": "Please show me how many employees working on different countries using a bar chart, could you list from high to low by the bars?", "output":"""Let's think step-by-step to generate the JSON for the bar chart:\nFirst, we need to specify the Vega-Lite schema:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json"}}\nNext, add a description for the chart:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "Number of Employees by Country"}}\nThen, specify the data source URL:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "Number of Employees by Country","data": {{"url": "https://example.com/data/employees.json"}}}}\nWe want a bar chart, so set the mark type to "bar":\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "Number of Employees by Country", "data": {{"url": "https://example.com/data/employees.json"}},"mark": "bar"}}\nNow define the x and y axis encodings:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "Number of Employees by Country","data": {{"url": "https://example.com/data/employees.json"}},"mark": "bar","encoding": {{"x": {{...}},"y": {{...}}}}}}\nSpecify the fields, types, and axis titles:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "Number of Employees by Country",  "data": {{"url": "https://example.com/data/employees.json"}},"mark": "bar","encoding": {{"x": {{"field": "employees", "type": "quantitative", "axis": {{"title": "Number of Employees"}}}},"y": {{"field": "country", "type": "nominal",  "axis": {{"title": "Country"}}}}}}}}\nFinally, sort y-axis in descending order:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "Number of Employees by Country","data": {{"url": "https://example.com/data/employees.json"}}, "mark": "bar","encoding": {{"x": {{"field": "employees", "type": "quantitative","axis": {{"title": "Number of Employees"}}}},"y": {{"field": "country", "type": "nominal", "axis": {{"title": "Country"}}, "sort": "-x"}}  }}\n"""},
            {"question": "plot a line chart on what is the average number of attendance at home games for each year?", "output": """Let's think step-by-step to generate the JSON for the line chart:\nFirst, specify the Vega-Lite schema:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json"}}\nAdd a description for the chart:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "Average Attendance at Home Games by Year"}}\n  Specify the data source URL:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "Average Attendance at Home Games by Year", "data": {{"url": "https://example.com/data/games.json"}}\nWe want a line chart, so set mark to "line":\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json",  "description": "Average Attendance at Home Games by Year","data": {{"url": "https://example.com/data/games.json"}},\n"mark": "line"}}Define the x-axis encoding:\n {{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "Average Attendance at Home Games by Year","data": {{"url": "https://example.com/data/games.json"}}, "mark": "line","encoding": {{"x": {{...}}\nSpecify the field, type, and axis title for x:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "Average Attendance at Home Games by Year", "data": {{"url": "https://example.com/data/games.json"}},"mark": "line",  "encoding": {{"x": {{"field": "year", "type": "ordinal", "axis": {{"title": "Year"}}}}\nDefine the y-axis encoding:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "Average Attendance at Home Games by Year","data": {{"url": "https://example.com/data/games.json"}},"mark": "line","encoding": {{"x": {{"field": "year", "type": "ordinal",  "axis": {{"title": "Year"}},"y": {{...}}}}\nSpecify the field, type, and axis title for y:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "Average Attendance at Home Games by Year","data": {{"url": "https://example.com/data/games.json"}},"mark": "line", "encoding": {{"x": {{"field": "year", "type": "ordinal","axis": {{"title": "Year"}},"y": {{"field": "average_attendance", "type": "quantitative","axis": {{"title": "Average Attendance"}}}}\n"""},
            {"question": "Pie Chart for Distribution of Expenses", "output":"""Let's think step-by-step to generate the JSON for the pie chart:\nFirst, specify the Vega-Lite schema:\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json"}}\nAdd a description for the chart: \n {{"$schema":"https://vega.github.io/schema/vega-lite/v4.json",\n"description":"Distribution of Expenses"}}\nSpecify the data source URL:\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json",\n"description":"Distribution of Expenses","data":{{"url":"https://example.com/data/expenses.json"}}}}We want a pie chart, so set mark to "arc":\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description":"Distribution of Expenses",  "data":{{"url":"https://example.com/data/expenses.json"}},"mark":"arc"}}\nDefine the theta encoding for the pie angles:\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description":"Distribution of Expenses","data":{{"url":"https://example.com/data/expenses.json"}},"mark":"arc","encoding":{{  "theta":{{...}}}}}}\nSpecify the field and type for theta:\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description":"Distribution of Expenses","data":{{"url":"https://example.com/data/expenses.json"}}, "mark":"arc","encoding":{{  "theta":{{"field":"amount","type":"quantitative"}}}}}}\nDefine the color encoding for categories:\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description":"Distribution of Expenses","data":{{"url":"https://example.com/data/expenses.json"}},"mark":"arc","encoding":{{  "theta":{{"field":"amount","type":"quantitative"}},  "color":{{...}}  }}}}\nSpecify the field and type for color:\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description":"Distribution of Expenses","data":{{"url":"https://example.com/data/expenses.json"}},"mark":"arc","encoding":{{  "theta":{{"field":"amount","type":"quantitative"}},  "color":{{"field":"category","type":"nominal"}}}}  }}}}}}}}\n"""},
            {"question": "Histogram for Exam Scores","output": """Let's think step-by-step to generate the JSON for the histogram:\nFirst, specify the Vega-Lite schema:\n  {{"$schema":"https://vega.github.io/schema/vega-lite/v4.json"}\nAdd a description for the chart:\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json",  "description":"Distribution of Exam Scores"}\nSpecify the data source URL:  \n {{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description":"Distribution of Exam Scores","data":{{"url":"https://example.com/data/exam-scores.json"}}\nWe want a histogram, so set mark to "bar":\n   {{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description":"Distribution of Exam Scores","data":{{"url":"https://example.com/data/exam-scores.json"}},"mark":"bar"}}\nDefine the x-axis encoding:\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description":"Distribution of Exam Scores","data":{{"url":"https://example.com/data/exam-scores.json"}},  "mark":"bar","encoding":{{  "x":{{...}}}}\nSpecify the bin, field, type, and axis title: \n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description":"Distribution of Exam Scores",  "data":{{"url":"https://example.com/data/exam-scores.json"}},"mark":"bar","encoding":{{  "x":{{"bin":true,"field":"data","type":"quantitative",       "axis":{{"title":"Score Range"}}}}\nDefine the y-axis encoding:\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json", "description":"Distribution of Exam Scores","data":{{"url":"https://example.com/data/exam-scores.json"}},"mark":"bar","encoding":{{  "x":{{...}},  "y":{{...}} }}}}\nSpecify the aggregate, type, and axis title:\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description":"Distribution of Exam Scores","data":{{"url":"https://example.com/data/exam-scores.json"}},"mark":"bar","encoding":{{  "x":{{...}},  "y":{{"aggregate":"count","type":"quantitative",  "axis":{{"title":"Frequency"}}}}\n"""},
            {"question": "Area Chart for Temperature Trends Over Time","output": """Let's think step-by-step to generate the JSON for the area chart:\nFirst, specify the Vega-Lite schema:\n{{{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json"}}\nAdd a description for the chart:\n{{{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description":"Temperature Trends Over Time"}}\nSpecify the data source URL:\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json",  "description":"Temperature Trends Over Time","data":{{"url":"https://example.com/data/Trends.json"}}}}\nWe want an area chart, so set mark to "area":\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description":"Temperature Trends Over Time","data":{{"url":"https://example.com/data/Trends.json"}},"mark":"area"}}\nDefine the x-axis encoding:\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description":"Temperature Trends Over Time", "data":{{"url":"https://example.com/data/Trends.json"}},"mark":"area","encoding":{{  "x":{{...}}}}}}\nSpecify the field, type, and axis title for x:\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description":"Temperature Trends Over Time","data":{{"url":"https://example.com/data/Trends.json"}},"mark":"area","encoding":{{  "x":{{"field":"year","type":"ordinal","axis":{{"title":"Year"}}}}}}}}\nDefine the y-axis encoding:\n {{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description":"Temperature Trends Over Time",  "data":{{"url":"https://example.com/data/Trends.json"}}, "mark":"area","encoding":{{  "x":{{...}},  "y":{{...}}}}\nSpecify the field, type, and axis title for y:\n{{"$schema":"https://vega.github.io/schema/vega-lite/v4.json","description":"Temperature Trends Over Time","data":{{"url":"https://example.com/data/Trends.json"}},"mark":"area","encoding":{{  "x":{{...}},  "y":{{"field":"temperature","type":"quantitative","axis":{{"title":"Temperature"}}  }}\n"""},
            {"question": "Annual Weather Heatmap", "output": """Let's think step-by-step to generate the JSON for the heatmap:\nFirst, specify the Vega-Lite schema:\n{{{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json"}}\nDefine the data source:\n{{{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json", "data": {{"url": "data/seattle-weather.csv"}}\nAdd a title:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","data": {{"url": "data/seattle-weather.csv"}},"title": "Daily Max Temperatures (C) in Seattle, WA"}}\nSpecify the configuration:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","data": {{"url": "data/seattle-weather.csv"}, "title": "...", "config": {{...}}\nSet the mark type to rectangle:\n {{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","data": {{"url": "data/seattle-weather.csv"}},"title": "...","config": {{...}}, "mark": "rect"}}\nDefine the x-axis encoding:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json",  "data": {{"url": "data/seattle-weather.csv"}},"title": "...","config": {{...}},"mark": "rect","encoding": {{"x": {{...}}\nAnd so on for y-axis, color encoding and complete the full JSON schema.\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","data": {{"url": "data/seattle-weather.csv"}},  "title": "Daily Max Temperatures (C) in Seattle, WA","config": {{  "view": {{"strokeWidth": 0, "step": 13}},  "axis": {{"domain": false}},"mark": "rect","encoding": {{  "x": {{    "field": "date",     "timeUnit": "date",     "type": "ordinal",    "title": "Day",    "axis": {{"labelAngle": 0, "format": "%e"}}  }},  "y": {{    "field": "date",    "timeUnit": "month",     "type": "ordinal",    "title": "Month"  }},  "color": {{    "field": "temp_max",    "aggregate": "max",    "type": "quantitative",    "legend": {{"title": null}}  }}}}\n"""},
            {"question": "Visualize the relationships between various measurements of penguin features (parallel coordinate plot)" , "output": """Let's think step-by-step to generate the JSON for the parallel coordinate plot:First, specify the Vega-Lite schema and add a description:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json", "description": "..."}}\nDefine the data source\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "...","data": {{"url": "data/penguins.json"}}}}\nSet the dimensions:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "...", "data": {{"url": "data/penguins.json"}},"width": 600,"height": 300}}\nAdd data transformations:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "...","data": {{"url": "data/penguins.json"}}, "width": 600,"height": 300,"transform": [  {{...}},   {{...}},  {{...}}]}}\nDefine the layers for axes, lines, and labels:\n {{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "...","data": {{"url": "data/penguins.json"}},"width": 600,  "height": 300,"transform": [...], "layer": [  {{...}},  {{...}},  {{...}}]}}\nAnd specify the configuration:\n{{"$schema": "https://vega.github.io/schema/vega-lite/v4.json","description": "...","data": {{"url": "data/penguins.json"}}, "width": 600,"height": 300,"transform": [...],"layer": [...],"config": {{...}}}}\nHere is the final Output:\n{{    "$schema": "https://vega.github.io/schema/vega-lite/v4.json",    "description": "Though Vega-Lite supports only one scale per axes, one can create a parallel coordinate plot by folding variables, using `joinaggregate` to normalize their values and using ticks and rules to manually create axes.",    "data": {{      "url": "data/penguins.json"    }},    "width": 600,    "height": 300,    "transform": [      {{"filter": "datum['Beak Length (mm)']"}},      {{"window": [{{"op": "count", "as": "index"}}]}},      {{"fold": ["Beak Length (mm)", "Beak Depth (mm)", "Flipper Length (mm)", "Body Mass (g)"]}},      {{        "joinaggregate": [          {{"op": "min", "field": "value", "as": "min"}},          {{"op": "max", "field": "value", "as": "max"}}        ],        "groupby": ["key"]      }},      {{        "calculate": "(datum.value - datum.min) / (datum.max-datum.min)",        "as": "norm_val"      }},      {{        "calculate": "(datum.min + datum.max) / 2",        "as": "mid"      }}    ],    "layer": [      {{        "mark": {{"type": "rule", "color": "#ccc"}},        "encoding": {{"detail": {{"aggregate": "count"}}, "x": {{"field": "key"}}}}      }},      {{        "mark": "line",        "encoding": {{          "color": {{"type": "nominal", "field": "Species"}},          "detail": {{"type": "nominal", "field": "index"}},          "opacity": {{"value": 0.3}},          "x": {{"type": "nominal", "field": "key"}},          "y": {{"type": "quantitative", "field": "norm_val", "axis": null}},          "tooltip": [            {{"type": "quantitative", "field": "Beak Length (mm)"}},            {{"type": "quantitative", "field": "Beak Depth (mm)"}},             {{"type": "quantitative", "field": "Flipper Length (mm)"}},            {{"type": "quantitative", "field": "Body Mass (g)"}}          ]        }}      }},      {{        "encoding": {{"x": {{"type": "nominal", "field": "key"}}, "y": {{"value": 0}}}},         "layer": [          {{            "mark": {{"type": "text", "style": "label"}},            "encoding": {{"text": {{"aggregate": "max", "field": "max"}}          }},          {{"mark": {{"type": "tick", "style": "tick", "size": 8, "color": "#ccc"}}}}        ]      }},              "encoding": {{"x": {{"type": "nominal", "field": "key"}}, "y": {{"value": 150}}}},        "layer": [          {{            "mark": {{"type": "text", "style": "label"}},            "encoding": {{"text": {{"aggregate": "min", "field": "mid"}}}}          }},          {{"mark": {{"type": "tick", "style": "tick", "size": 8, "color": "#ccc"}}}}          ]      }},      {{        "encoding": {{"x": {{"type": "nominal", "field": "key"}}, "y": {{"value": 300}}}},        "layer": [          {{            "mark": {{"type": "text", "style": "label"}},            "encoding": {{"text": {{"aggregate": "min", "field": "min"}}}}          }},          {{"mark": {{"type": "tick", "style": "tick", "size": 8, "color": "#ccc"}}}}        ]      }}    ],    "config": {{      "axisX": {{        "domain": false,        "labelAngle": 0,        "tickColor": "#ccc",        "title": null      }},      "view": {{"stroke": null}},      "style": {{        "label": {{"baseline": "middle", "align": "right", "dx": -5}},        "tick": {{"orient": "horizontal"}}      }}    }}  }}\n"""}
        ]
        self.llm = initialize_evllm(model_id= self.model_id, temperature=0.3)
        self.prompt = PromptTemplate(input_variables=["question", "output"], template= """Here are some example:\nQuestion: {question}\nVEGALITE JSON: {output}""")
        self.selector = SemanticSimilarityExampleSelector.from_examples(self.shots, OpenAIEmbeddings(), FAISS, k=1)
        self.FewShotPrompt = FewShotPromptTemplate(
            example_selector=self.selector,
            example_prompt=self.prompt,
            prefix="""Generate Vegalite JSON Specification for given Query.""",
            suffix="""Previous Conversation:{chat_history}\nData:\n{context}\nInput: {question}\nVegaLite-JSON:""",
            input_variables=["context","question","chat_history"],
        )


        self.data_url = None
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
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

            
            vis_chain = ConversationalRetrievalChain.from_llm(
                self.llm,
                retriever=csv_retriever,
                combine_docs_chain_kwargs={"prompt": self.FewShotPrompt},
                memory=memory,
                verbose=True
            )
            result = vis_chain({"question": input})
            result = result["answer"]
            return result
        except(SyntaxError, ValueError) as e:
            print(f"Error in visQA chain func: {str(e)}")


    def generate(self, query, dataFile, truth):
        try:
            predicted = self.visQA_chain(dataFile,query)
            try:
                # pred = predicted
                pred = predicted.replace('true', 'True')
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
            # if index == 50:
            #     break
            query = row['Utterance Set']
            vlSpec_output = row['VegaLiteSpec']
            Datafile = row['dataset'].lower()
            self.generate(query, Datafile, vlSpec_output)
        return "Evaluation Process Completed!!!"
