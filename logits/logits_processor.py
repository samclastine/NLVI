import torch
from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data
from lmformatenforcer import JsonSchemaParser
import vllm
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union

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


def delete_llm(model_initialized_var):
    """
    Function to delete Long-Long Memory (LLM) by emptying the CUDA memory cache.
    """
    del model_initialized_var
    torch.cuda.empty_cache()

class ProcessorLogit:
    def __init__(self, model_id):
        """
        Initialize ProcessorLogit with the provided model ID.
        """
        self.model_id = model_id
        llm = vllm.LLM(model=model_id)
        self.tokenizer_data = build_vllm_token_enforcer_tokenizer_data(llm)
        del llm
        torch.cuda.empty_cache()
        self.logits_processor = build_vllm_logits_processor(self.tokenizer_data, JsonSchemaParser(VegaLiteSchema.schema()))




def create_processor_logit(model_id):
    """
    Function to create and return an instance of ProcessorLogit.
    """
    return ProcessorLogit(model_id)