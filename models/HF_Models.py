
from VLLM import EVLLM
from logits import create_processor_logit
import warnings

warnings.filterwarnings('ignore')
class EVLLMInitializer:
    def __init__(self, model_id, temperature=0.3):
        """
        Initializes the EVLLMInitializer with the model ID and optional temperature setting.
        
        :param model_id: str, identifier for the model
        :param temperature: float, temperature setting for the model, default is 0.3
        """
        self.model_id = model_id
        self.temperature = temperature
        self.llm = None  # This will hold the instance of EVLLM

    def initialize_model(self):
        """
        Initializes the EVLLM model with the given configuration.
        
        :return: EVLLM, an instance of the EVLLM class configured with the provided settings
        """
        # Create the logits processor using the utility function
        processor = create_processor_logit(model_id=self.model_id)

        # Initialize the EVLLM with specific configurations
        self.llm = EVLLM(
            model=self.model_id,
            trust_remote_code=True,  # Mandatory for models from Hugging Face, etc.
            max_new_tokens=1024,
            top_k=10,
            top_p=0.95,
            temperature=self.temperature,
            logits_processors=[processor['logits_processor']]  # Integrating the logits processor
        )
        return self.llm

    def get_model(self):
        """
        Returns the initialized EVLLM model. Initializes the model if not already done.
        
        :return: EVLLM, the initialized model instance
        """
        if self.llm is None:
            return self.initialize_model()
        return self.llm

# # Example usage:
# # initializer = EVLLMInitializer("model_123")
# # llm_instance = initializer.get_model()

# def initialize_evllm(temperature:0.3, model_id):
#     processor = create_processor_logit(model_id=model_id)
#     # Initialize the EVLLM with specific configurations
#     llm = EVLLM(
#         model=model_id,
#         trust_remote_code=True,  # Mandatory for models from Hugging Face, etc.
#         max_new_tokens=1024,
#         top_k=10,
#         top_p=0.95,
#         temperature=temperature,
#         logits_processors=[processor['logits_processor']]  # Integrating your logits processor
#     )

#     return llm

# Example usage:
# llm_instance = initialize_evllm(0.3, "model_123")
