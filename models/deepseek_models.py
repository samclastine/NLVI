from langchain_openai import ChatOpenAI




def initialize_openai_model(model_id, temperature):

    model = ChatOpenAI(
                model=model_id,
                openai_api_key="sk-7568cf9f229345989115cc25dedfa3cd",
                openai_api_base='https://api.deepseek.com/v1',
                temperature=temperature,
                max_tokens=4012)
    return model