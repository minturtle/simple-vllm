from simple_vllm.core.LLM import LLM
import pytest

def test_create_LLM_object():
  model_id = "minseok/mymodel-1b"
  tokenizer_id = "minseok/mymodel-1b(tokenizer_test)"
  llm = LLM(model_name = model_id, tokenizer_name = tokenizer_id)
  assert llm.model_name == model_id
  assert llm.tokenizer_name == tokenizer_id


def test_generate_input_prompts_many_types():
  model_id = "minseok/mymodel-1b"
  tokenizer_id = "minseok/mymodel-1b(tokenizer_test)"
  llm = LLM(model_name = model_id, tokenizer_name = tokenizer_id)

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