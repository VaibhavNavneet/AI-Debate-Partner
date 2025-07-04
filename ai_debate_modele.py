# -*- coding: utf-8 -*-
"""ai_debate_modele.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11Vt40FTiVB7pqpTlfNJA4xQoKNt5AR06
"""

!pip install jupyterlab ipykernel ipywidgets datasets  torch torchvision torchaudio transformers peft evaluate

# AI Debate Partner - Argument Generation (Fixed Loss Issue)
# Fixes ValueError: The model did not return a loss

## Step 1: Import Libraries
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import pandas as pd
import torch
import numpy as np

## Step 2: Load and Preprocess Dataset
df = pd.read_csv('/content/train.csv')
df = df[['topic', 'argument', 'stance_WA']].dropna()
df['stance_label'] = df['stance_WA'].map({1: "pro", -1: "con"})
df['text'] = "Topic: " + df['topic'] + "; Stance: " + df['stance_label'] + "; Argument: " + df['argument'] + tokenizer.eos_token

# Create Dataset object
dataset = Dataset.from_pandas(df[['text']])

## Step 3: Tokenization
model_checkpoint = 'meta-llama/Llama-2-7b'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    # Create labels by copying input_ids
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

## Step 4: Model Setup
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
model.config.use_cache = False  # Disable cache to prevent returning past_key_values

## Step 5: PEFT Configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

## Step 6: Training Setup
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling
)

training_args = TrainingArguments(
    output_dir="ai-debate-generator",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=500,
    report_to="none",
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

## Step 7: Training
trainer.train()  # Should now compute loss

## Step 8: Argument Generation Function
def generate_argument(topic: str, stance: str):
    stance = stance.lower()
    if stance not in ["pro", "con"]:
        raise ValueError("Stance must be 'pro' or 'con'")

    prompt = f"Topic: {topic}; Stance: {stance}; Argument:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Argument:" in generated_text:
        return generated_text.split("Argument:")[1].strip()
    return generated_text

## Step 9: Interactive Usage
if __name__ == "__main__":
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    user_topic = input("Enter debate topic: ")
    user_stance = input("Enter your stance (pro/con): ")

    try:
        argument = generate_argument(user_topic, user_stance)
        print("\nGenerated Argument:")
        print("-------------------")
        print(argument)
    except Exception as e:
        print(f"\nError: {str(e)}")

## Step 10: Save Model
model.save_pretrained("ai-debate-generator-model")
tokenizer.save_pretrained("ai-debate-generator-tokenizer")



# Zip the folder (e.g., "my_model") into "my_model.zip"
shutil.make_archive("tokenizer", 'zip', "/content/ai-debate-generator-tokenizer")

