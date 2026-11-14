import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.arithmetic import is_correct, make_dataset, reward_function
from grpx.loop import TrainingConfig, train
from grpx.backends.huggingface import HuggingFaceBackend
from grpx.environment import SingleTurnEnvironment


model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
model = get_peft_model(
    model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    )
)
model.print_trainable_parameters()


backend = HuggingFaceBackend(model, tokenizer)
train(model, tokenizer, backend, make_dataset(2000, seed=0),
      lambda: SingleTurnEnvironment(reward_function, is_correct), TrainingConfig(),
      evaluation_dataset=make_dataset(200, seed=1))
