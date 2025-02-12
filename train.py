from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

model_name = "meta-llama/Llama-3.2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("tatsu-lab/alpaca")  

training_args = TrainingArguments(
    output_dir="./llama-finetuned",
    per_device_train_batch_size=4,
    save_total_limit=2,
    num_train_epochs=3,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()
