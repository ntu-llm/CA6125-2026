import re
import random
import numpy as np
import torch
import pandas as pd

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward

import torch
import trl
import transformers

print("torch version:", torch.__version__)
print("trl version:", trl.__version__)
print("transformers version:", transformers.__version__)


# Fix randomness for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# DeepMath-103K provides "prompt" and "solution"
dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

# Use a small subset for a notebook demo
dataset = dataset.select(range(300))

# Create train / eval split
split = dataset.train_test_split(test_size=50, seed=seed)
train_dataset = split["train"]
eval_dataset = split["test"]

print("Train size:", len(train_dataset))
print("Eval size :", len(eval_dataset))
print("\nExample prompt:\n")
print(train_dataset[0]["prompt"])
print("\nExample solution:\n")
print(train_dataset[0]["solution"][:300])


model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Model before GRPO
model_before = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Keep the config small enough for a notebook demo
training_args = GRPOConfig(
    output_dir="./grpo_demo",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-6,
    num_generations=4,
    max_completion_length=64,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    fp16=True,
)

trainer = GRPOTrainer(
    model=model_name,
    args=training_args,
    reward_funcs=accuracy_reward,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

trainer.train()

# Trained model after GRPO
model_after = trainer.model

def generate_completion(model, prompt, max_new_tokens=128):
    """Generate one deterministic completion for fair before/after comparison."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # deterministic evaluation
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_boxed(text):
    """Extract the final answer inside \\boxed{} if present."""
    match = re.search(r"\\boxed\{(.*?)\}", text)
    return match.group(1).strip() if match else None


def extract_boxed_from_solution(solution_text):
    """Extract the gold boxed answer from the reference solution."""
    return extract_boxed(solution_text)


def compute_reward_from_reference(completion, gold_solution):
    """
    Paper-style simple reward:
    1.0 if the boxed answer matches the reference boxed answer, else 0.0
    """
    pred = extract_boxed(completion)
    gold = extract_boxed_from_solution(gold_solution)
    if pred is None or gold is None:
        return 0.0
    return 1.0 if pred == gold else 0.0


def evaluate_model(model, eval_dataset, n_examples=20):
    """
    Standard notebook-style evaluation:
    1. Reward: mean reward on eval set
    2. Task performance: exact-match accuracy
    3. Generation quality: answer extraction rate
    """
    rewards = []
    correct = 0
    valid_boxed = 0
    rows = []

    subset = eval_dataset.select(range(min(n_examples, len(eval_dataset))))

    for example in subset:
        prompt = example["prompt"]
        gold_solution = example["solution"]
        gold_answer = extract_boxed_from_solution(gold_solution)

        completion = generate_completion(model, prompt)
        pred_answer = extract_boxed(completion)

        reward = compute_reward_from_reference(completion, gold_solution)
        rewards.append(reward)

        if pred_answer is not None:
            valid_boxed += 1
        if pred_answer is not None and gold_answer is not None and pred_answer == gold_answer:
            correct += 1

        rows.append({
            "prompt": prompt,
            "gold_answer": gold_answer,
            "completion": completion,
            "pred_answer": pred_answer,
            "reward": reward,
        })

    mean_reward = float(np.mean(rewards))
    accuracy = correct / len(subset)
    boxed_rate = valid_boxed / len(subset)

    metrics = {
        "mean_reward": mean_reward,
        "task_accuracy": accuracy,
        "boxed_answer_rate": boxed_rate,
    }
    return metrics, rows



before_metrics, before_rows = evaluate_model(model_before, eval_dataset, n_examples=20)
after_metrics, after_rows = evaluate_model(model_after, eval_dataset, n_examples=20)

print("=== Before GRPO ===")
for k, v in before_metrics.items():
    print(f"{k}: {v:.4f}")

print("\n=== After GRPO ===")
for k, v in after_metrics.items():
    print(f"{k}: {v:.4f}")


print("=== Qualitative Comparison ===\n")

for i in range(5):
    print(f"Example {i+1}")
    print("-" * 80)
    print("Prompt:")
    print(before_rows[i]["prompt"][:300], "...\n")

    print("Gold answer:")
    print(before_rows[i]["gold_answer"], "\n")

    print("Before GRPO completion:")
    print(before_rows[i]["completion"][:500], "\n")
    print("Before predicted answer:", before_rows[i]["pred_answer"])
    print("Before reward:", before_rows[i]["reward"], "\n")

    print("After GRPO completion:")
    print(after_rows[i]["completion"][:500], "\n")
    print("After predicted answer:", after_rows[i]["pred_answer"])
    print("After reward:", after_rows[i]["reward"])
    print("=" * 80, "\n")



summary = {
    "Metric": ["Mean Reward", "Task Accuracy", "Boxed Answer Rate"],
    "Before GRPO": [
        before_metrics["mean_reward"],
        before_metrics["task_accuracy"],
        before_metrics["boxed_answer_rate"],
    ],
    "After GRPO": [
        after_metrics["mean_reward"],
        after_metrics["task_accuracy"],
        after_metrics["boxed_answer_rate"],
    ],
}

pd.DataFrame(summary)
