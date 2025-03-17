from typing import Union
import torch

class LLM:
  def __init__(self, model_name : str, tokenizer_name : str) -> None:
    self.model_name = model_name
    self.tokenizer_name = tokenizer_name
    if not torch.cuda.is_available():
      raise Exception("현재 LLM 클래스는 CUDA 환경에서만 동작합니다.")

  def generate(self, prompts : Union[str, list[str]]):
    if not isinstance(prompts, (str, list)):
        raise TypeError(f"prompts must be of type str or list[str], got {type(prompts).__name__}")
    
    if isinstance(prompts, list) and not all(isinstance(p, str) for p in prompts):
        raise TypeError("All elements in prompts list must be of type str")