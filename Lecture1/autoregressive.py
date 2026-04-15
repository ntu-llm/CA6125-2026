from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load model
model = AutoModelForCausalLM.from_pretrained(model_name)

# optional: test generation
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
