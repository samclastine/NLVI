from VLLM import EVLLM
from logits import create_logits_processor
import config
config.init() 
# from logits_processor import create_logits_processor

# print(create_logits_processor)


def initialize_evllm(temperature:0.3, model_id):
    processor = create_logits_processor(model_id=model_id)
    # Initialize the EVLLM with specific configurations
    llm = EVLLM(
        model=model_id,
        trust_remote_code=True,  # Mandatory for models from Hugging Face, etc.
        max_new_tokens=4096,
        top_k=10,
        top_p=0.95,
        temperature=temperature,
        download_dir=config.model_dir,
        logits_processors=[processor],  # Integrating your logits processor
        gpu_memory_utilization=0.9,
        vllm_kwargs={"max_model_len": 4096}
    )

    return llm

# Example usage:
# llm_instance = initialize_evllm(0.3, "model_123")
