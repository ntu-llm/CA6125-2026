import random
import numpy as np
import torch

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import GRPOConfig, GRPOTrainer

# Fix random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Use a small instruct model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load a copy of the base model for before/after comparison
model_before = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

print("Loaded model:", model_name)
print("Pad token id:", tokenizer.pad_token_id)

# Use a unified prompt template
# The goal is clear: generate one short positive movie review sentence in natural English
train_prompts = [
    "Write one short positive movie review sentence in natural English.",
] * 64

train_ds = Dataset.from_dict({"prompt": train_prompts})

print("Training set size:", len(train_ds))
print("Example prompt:")
print(train_ds[0]["prompt"])

# Reward model: sentiment classifier
reward_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1
)

print("Reward model loaded.")

# Reward only looks at the completion, not the prompt
# This avoids the prompt dominating the sentiment score
def sentiment_reward(completions):
    rewards = []
    for completion in completions:
        text = completion.strip()
        pred = reward_model(text)[0]
        score = pred["score"] if pred["label"] == "POSITIVE" else 1.0 - pred["score"]
        rewards.append(float(score))
    return rewards

# Stable generation helper for before/after comparison
# We keep temperature and top_p fixed as requested
def generate_text(model, prompt, max_new_tokens=24):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# Reward helper for single string
def get_reward(text):
    pred = reward_model(text)[0]
    return pred["score"] if pred["label"] == "POSITIVE" else 1.0 - pred["score"]

print("=" * 60)
print("BEFORE TRAINING: SAMPLE GENERATIONS")
print("=" * 60)

test_prompt = "Write one short positive movie review sentence in natural English."

before_examples = []
for i in range(5):
    completion = generate_text(model_before, test_prompt)
    reward = get_reward(completion)
    before_examples.append((completion, reward))
    print(f"\nExample {i+1}")
    print("Prompt:", test_prompt)
    print("Completion:", completion)
    print("Reward:", reward)

# Evaluate average reward before training
def eval_model(model, prompt, n=20):
    scores = []
    outputs = []
    for _ in range(n):
        completion = generate_text(model, prompt)
        score = get_reward(completion)
        scores.append(score)
        outputs.append(completion)
    return float(np.mean(scores)), outputs

eval_prompt = "Write one short positive movie review sentence in natural English."

before_score, before_outputs = eval_model(model_before, eval_prompt, n=20)

print("Average reward before GRPO:", before_score)

# Configure GRPO
training_args = GRPOConfig(
    output_dir="./grpo_results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=1,
    num_generations=4,
    max_completion_length=24,
    temperature=0.7,
    top_p=0.9,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    bf16=False,
    fp16=torch.cuda.is_available(),
)

# Build trainer and train
trainer = GRPOTrainer(
    model=model_name,
    args=training_args,
    reward_funcs=sentiment_reward,
    train_dataset=train_ds,
    processing_class=tokenizer,
)

print("=" * 60)
print("START GRPO TRAINING")
print("=" * 60)

trainer.train()

print("=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

model_after = trainer.model

print("=" * 60)
print("AFTER TRAINING: SAMPLE GENERATIONS")
print("=" * 60)

after_examples = []
for i in range(5):
    completion = generate_text(model_after, test_prompt)
    reward = get_reward(completion)
    after_examples.append((completion, reward))
    print(f"\nExample {i+1}")
    print("Prompt:", test_prompt)
    print("Completion:", completion)
    print("Reward:", reward)

# Evaluate average reward after training
after_score, after_outputs = eval_model(model_after, eval_prompt, n=20)

print("Average reward after GRPO:", after_score)

print("=" * 60)
print("AVERAGE REWARD COMPARISON")
print("=" * 60)
print("Before GRPO:", before_score)
print("After GRPO :", after_score)

print("=" * 60)
print("QUALITATIVE COMPARISON")
print("=" * 60)

for i in range(5):
    print(f"\nPair {i+1}")
    print("Before:", before_outputs[i])
    print("After :", after_outputs[i])