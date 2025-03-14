from typing import Union

class LLM:
  def __init__(self, model_name : str, tokenizer_name : str) -> None:
    self.model_name = model_name
    self.tokenizer_name = tokenizer_name


  def generate(self, prompts : Union[str, list[str]]):
    if not isinstance(prompts, (str, list)):
        raise TypeError(f"prompts must be of type str or list[str], got {type(prompts).__name__}")
    
    if isinstance(prompts, list) and not all(isinstance(p, str) for p in prompts):
        raise TypeError("All elements in prompts list must be of type str")