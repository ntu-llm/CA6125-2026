import random
import re
from statistics import mean

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


# Reproducibility helps keep the classroom demo stable.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# The reward is intentionally simple and transparent:
# RL should learn to complete a review clause with clearly positive language.
positive_words = [
    "amazing", "beautiful", "brilliant", "charming", "delightful", "excellent",
    "fantastic", "graceful", "heartfelt", "impressive", "memorable", "moving",
    "outstanding", "powerful", "stunning", "wonderful",
]
negative_words = [
    "awkward", "bad", "bland", "boring", "confused", "disappointing",
    "dull", "flat", "forgettable", "generic", "messy", "predictable",
    "slow", "weak",
]


def build_user_prompt(title, description):
    return (
        f"Movie: {title}\n"
        f"Description: {description}\n"
        "Complete this review in exactly one short English sentence.\n"
        "Start with: The movie was"
    )


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def format_chat_prompt(user_prompt):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise film critic. "
                "Reply with exactly one short English review sentence."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


train_examples = [
    {
        "title": "Echoes of the Sea",
        "description": "A frustrated pianist returns to a quiet coastal town and reconnects with family.",
    },
    {
        "title": "Last Train at Midnight",
        "description": "Three strangers exchange secrets on an overnight train journey.",
    },
    {
        "title": "The Glass Garden",
        "description": "A broken family faces old wounds during a long stormy night.",
    },
    {
        "title": "Cloud Hotel",
        "description": "Guests at a remote mountain hotel carry regrets they cannot easily say aloud.",
    },
]

train_prompts = [
    format_chat_prompt(build_user_prompt(item["title"], item["description"]))
    for item in train_examples
    for _ in range(24)
]
train_dataset = Dataset.from_dict({"prompt": train_prompts})


model_before = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)


def reward_from_text(text):
    text = text.strip()
    text_lower = text.lower()

    word_count = len(re.findall(r"[A-Za-z']+", text))
    ascii_ratio = sum(ord(ch) < 128 for ch in text) / max(len(text), 1)
    sentence_count = sum(text.count(mark) for mark in [".", "!", "?"])
    positive_hits = sum(word in text_lower for word in positive_words)
    negative_hits = sum(word in text_lower for word in negative_words)
    bad_symbols = sum(ch in "{}[]<>\\/|_*#`$@" for ch in text)
    starts_correctly = text_lower.startswith("the movie was")

    reward = 0.0
    reward += positive_hits * 2.0
    reward -= negative_hits * 2.0

    if starts_correctly:
        reward += 3.0
    else:
        reward -= 4.0

    if 8 <= word_count <= 18:
        reward += 2.0
    else:
        reward -= 2.0

    if sentence_count == 1 and text.endswith((".", "!", "?")):
        reward += 1.5
    else:
        reward -= 1.5

    if ascii_ratio > 0.99:
        reward += 1.5
    else:
        reward -= 4.0

    if bad_symbols == 0:
        reward += 1.0
    else:
        reward -= min(4.0, bad_symbols * 0.5)

    return {
        "reward": float(reward),
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "word_count": word_count,
    }


def positive_review_reward(prompts, completions, **kwargs):
    return [reward_from_text(completion)["reward"] for completion in completions]


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


def evaluate_model(model, prompts, samples_per_prompt=3):
    rows = []
    rewards = []

    for prompt_item in prompts:
        for _ in range(samples_per_prompt):
            text = generate_text(model, prompt_item["model_prompt"])
            metrics = reward_from_text(text)
            rows.append(
                {
                    "prompt": prompt_item["display_prompt"],
                    "output": text,
                    **metrics,
                }
            )
            rewards.append(metrics["reward"])

    return rows, float(mean(rewards))


eval_prompts = [
    {
        "display_prompt": build_user_prompt(
            "Echoes of the Sea",
            "A frustrated pianist returns to a quiet coastal town and reconnects with family.",
        ),
        "model_prompt": format_chat_prompt(
            build_user_prompt(
                "Echoes of the Sea",
                "A frustrated pianist returns to a quiet coastal town and reconnects with family.",
            )
        ),
    },
    {
        "display_prompt": build_user_prompt(
            "Last Train at Midnight",
            "Three strangers exchange secrets on an overnight train journey.",
        ),
        "model_prompt": format_chat_prompt(
            build_user_prompt(
                "Last Train at Midnight",
                "Three strangers exchange secrets on an overnight train journey.",
            )
        ),
    },
    {
        "display_prompt": build_user_prompt(
            "Summer Departure",
            "A group of college friends face love, friendship, and uncertainty during one final trip.",
        ),
        "model_prompt": format_chat_prompt(
            build_user_prompt(
                "Summer Departure",
                "A group of college friends face love, friendship, and uncertainty during one final trip.",
            )
        ),
    },
]


print("=" * 80)
print("BEFORE RL")
print("=" * 80)
before_rows, before_mean_reward = evaluate_model(model_before, eval_prompts)

for i, row in enumerate(before_rows[:6], start=1):
    print(f"\n[Before {i}]")
    print("Prompt:")
    print(row["prompt"])
    print("Output:")
    print(row["output"])
    print(
        f"Reward: {row['reward']:.2f} | "
        f"positive_words: {row['positive_hits']} | "
        f"negative_words: {row['negative_hits']} | "
        f"words: {row['word_count']}"
    )

print(f"\nAverage reward before RL: {before_mean_reward:.4f}")


training_args = GRPOConfig(
    output_dir="./grpo_positive_movie_reviews",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
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

trainer = GRPOTrainer(
    model=model_name,
    args=training_args,
    reward_funcs=positive_review_reward,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

print("\n" + "=" * 80)
print("START RL TRAINING")
print("=" * 80)
trainer.train()

final_model_dir = "./grpo_positive_movie_reviews/final_model"
trainer.model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

model_after = AutoModelForCausalLM.from_pretrained(
    final_model_dir,
    torch_dtype="auto",
    device_map="auto",
)
model_after.eval()

print("\n" + "=" * 80)
print("AFTER RL")
print("=" * 80)
after_rows, after_mean_reward = evaluate_model(model_after, eval_prompts)

for i, row in enumerate(after_rows[:6], start=1):
    print(f"\n[After {i}]")
    print("Prompt:")
    print(row["prompt"])
    print("Output:")
    print(row["output"])
    print(
        f"Reward: {row['reward']:.2f} | "
        f"positive_words: {row['positive_hits']} | "
        f"negative_words: {row['negative_hits']} | "
        f"words: {row['word_count']}"
    )

print(f"\nAverage reward after RL: {after_mean_reward:.4f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Average reward before RL: {before_mean_reward:.4f}")
print(f"Average reward after  RL: {after_mean_reward:.4f}")
print("\nWhat RL learned in this demo:")
print("1. Keep the output as one short English review sentence.")
print("2. Start the review in the requested form: The movie was ...")
print("3. Use clearly more positive review words after RL.")
