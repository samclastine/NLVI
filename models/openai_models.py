from langchain_community.chat_models import ChatOpenAI


def initialize_openai_model(model_id, temperature):
    model =  ChatOpenAI(model_name=model_id, temperature=temperature)
    return model