from langchain_openai import OpenAI


def initialize_openai_model(model_id):
    model =  OpenAI(model_name=model_id)
    return model