from openai import OpenAI
from langchain.vectorstores import FAISS
import urllib

from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings



class GPTEvaluator:
    def __init__(self, engine='gpt-4-turbo'):
        """
        Initialize the GPTRetriever with OpenAI API key, engine version, and FAISS index.
        """
        self.engine = engine


    def retrieve_documents(self, query, docName):
        """
        Retrieve relevant documents from the FAISS index for a given query.
        """
        data_url = 'https://raw.githubusercontent.com/nl4dv/nl4dv/master/examples/assets/data/' + docName

        
        if docName == "superstore":
            data_url = "https://raw.githubusercontent.com/nl4dv/nl4dv/master/examples/assets/data/" + docName + ".csv"
        else:
            data_url = "https://raw.githubusercontent.com/nlvcorpus/nlvcorpus.github.io/main/datasets/" + docName + ".csv"

        urllib.request.urlretrieve(data_url, docName)
        csv_loader = CSVLoader(file_path=docName)
        csv_data = csv_loader.load()
        csv_text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0
        )
        csv_docs = csv_text_splitter.split_documents(csv_data)
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(csv_docs, embeddings)
        retriever = db.as_retriever()
        docs = retriever.invoke(query)
        return docs

    def custom_prompt(self, input_text, document, pred):
        """
        Create a custom prompt with the given input and document information.
        """
        # Fetch the documents using the document_ids and integrate them into the prompt.
        # This is a placeholder. Replace it with your actual document fetching and formatting logic.
        documents = [f"{doc_id}" for doc_id in document]
        context = " ".join(documents)
        print(context)
        evaluation_template = f"""\n 'I am an expert in evaluating Vega-Lite JSON specifications.\
            My work involves meticulously analyzing JSON code based on specific queries and\
            criteria to assess its effectiveness and correctness.\
            I examine various aspects of the JSON, such as its structure, syntax, and \
            I should evaluatate is it a valid vegalite json\
            I provide an overall subjective score that reflects the quality of the JSON in terms of its adherence to best \
            practices, efficiency in data representation \
            This score helps in identifying areas of improvement and ensures that the Vega-Lite JSON meets the highest standards of data visualization. I should only provide the score (0 or 1) based on given criteria, for example the final output will be {{'Question':'.....', 'Score':..}} \n And i will not give any explanation.' \n Data: {context}\n Question: {input_text}\n Vega-lite Json: {pred}\n Score:"""
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": evaluation_template
            },
        ]
        return PROMPT_MESSAGES
    def run(self, input_text, docName, pred):
        """
        Run the complete retrieval and evaluation process: retrieve documents, generate prompt, and query GPT.
        """
        document_ids = self.retrieve_documents(input_text, docName)
        prompt = self.custom_prompt(input_text, document_ids, pred)
        client = OpenAI()
        response = client.chat.completions.create(model="gpt-4-0125-preview", messages=prompt)
        return response

evaluator = GPTEvaluator()
