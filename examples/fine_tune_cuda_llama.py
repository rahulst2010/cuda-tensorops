import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, TrainingArguments, Trainer
from cuda_llama import load_cuda_llama
from lora_integration import CUDALoraConfig

# Configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DATASET_NAME = "tatsu-lab/alpaca"
LORA_RANK = 8
LORA_ALPHA = 16

# Load components
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
model = load_cuda_llama()
dataset = load_dataset(DATASET_NAME)

def preprocess(examples):
    return tokenizer(
        [f"{inst} {inp}" for inst, inp in zip(examples['instruction'], examples['input'])],
        truncation=True, padding="max_length", max_length=512
    )

dataset = dataset.map(preprocess, batched=True)

# Configure LoRA
lora_config = CUDALoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

# Apply LoRA to CUDA-optimized model
model.enable_input_require_grads()
model = lora_config.create_model(model)

# Training setup
training_args = TrainingArguments(
    output_dir="./cuda_llama_finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    logging_steps=50,
    num_train_epochs=1,
    fp16=True,
    optim="adamw_torch",
    save_strategy="epoch",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data])}
)

# Start training
trainer.train()
trainer.save_model("./cuda_llama_finetuned/final")
