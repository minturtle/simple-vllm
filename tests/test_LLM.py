from simple_vllm.core.LLM import LLM
from unittest.mock import patch
import pytest


def test_create_LLM_object():
  model_id = "minseok/mymodel-1b"
  tokenizer_id = "minseok/mymodel-1b(tokenizer_test)"
  llm = LLM(model_name = model_id, tokenizer_name = tokenizer_id)
  assert llm.model_name == model_id
  assert llm.tokenizer_name == tokenizer_id


def test_generate_input_prompts_many_types():
  llm = generate_llm()

  try:
    llm.generate("test prompt")
  except Exception:
    pytest.fail("str은 입력이 가능해야 합니다.")

  try:
    llm.generate(["test prompt1, test prompt2"])
  except Exception:
    pytest.fail("list[str]은 입력이 가능해야 합니다.")

  with pytest.raises(TypeError):
    llm.generate(123123)

def test_LLM_runnable_CUDA_env():
  with patch('torch.cuda.is_available', return_value=True):
    try:
      generate_llm()
    except Exception:
      pytest.fail("CUDA환경에서는 동작 가능해야 합니다.")

def test_LLM_unrunnable_CPU_env():
 with patch('torch.cuda.is_available', return_value=False):
    with pytest.raises(Exception):
      generate_llm()


def generate_llm(
  model_name = "minseok/mymodel-1b",
  tokenizer_name = "minseok/mymodel-1b(tokenizer_test)"
):
  return LLM(model_name = model_name, tokenizer_name = tokenizer_name)
