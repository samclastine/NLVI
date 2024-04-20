import torch
from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data
from lmformatenforcer import JsonSchemaParser
import vllm
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import warnings
import gc
import config

config.init()
# Suppress warnings, if necessary
warnings.filterwarnings('ignore')

# Define the schema classes using Pydantic
class DataSource(BaseModel):
    url: str

class Mark(BaseModel):
    type: str
    tooltip: Optional[Any] = None
    filled: Optional[bool] = None
    opacity: Optional[float] = None

class Axis(BaseModel):
    title: Optional[str] = None
    format: Optional[str] = None
    labels: Optional[bool] = None
    ticks: Optional[bool] = None

class EncodingChannel(BaseModel):
    field: str
    type: str
    bin: Optional[bool] = None
    timeUnit: Optional[str] = None
    aggregate: Optional[str] = None
    axis: Optional[Axis] = None
    sort: Optional[Dict[str, Any]] = None
    scale: Optional[Dict[str, Any]] = None
    legend: Optional[Dict[str, Any]] = None

class Encoding(BaseModel):
    x: Optional[EncodingChannel] = None
    y: Optional[EncodingChannel] = None
    color: Optional[EncodingChannel] = None
    column: Optional[EncodingChannel] = None

class Transform(BaseModel):
    pass

class VegaLiteSchema(BaseModel):
    description: Optional[str] = None
    data: DataSource
    mark: Mark
    encoding: Encoding
    transform: Optional[List[Transform]] = None

# Function to delete model and clear CUDA cache
def delete_llm(model_initialized_var):
    del model_initialized_var
    torch.cuda.empty_cache()

# Function to create and return the logits processor
def create_logits_processor(model_id):
    """
    Create and return the logits processor using the given model ID.
    """
    # Initialize the LLM with the given model ID
    llm = vllm.LLM(model=model_id, download_dir=config.model_dir, max_model_len=4096)
    
    # Build the tokenizer data for enforcing token constraints
    tokenizer_data = build_vllm_token_enforcer_tokenizer_data(llm)

    # Delete the LLM instance to free up resources
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("Successfully deleted the llm and free the GPU memory!")

    # Build the logits processor using the tokenizer data and a JSON schema parser
    logits_processor = build_vllm_logits_processor(tokenizer_data, JsonSchemaParser(VegaLiteSchema.schema()))

    return logits_processor

# This function can now be used to get a logits processor which can then be integrated into further processing steps.
