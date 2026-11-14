import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.guess_number import GuessNumberEnvironment, make_dataset
from grpx.loop import TrainingConfig, train
from grpx.backends.huggingface import HuggingFaceBackend


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
      lambda: GuessNumberEnvironment(reveal_interval=True), TrainingConfig(max_steps=6),
      evaluation_dataset=make_dataset(50, seed=1))
