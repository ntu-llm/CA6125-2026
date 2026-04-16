
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

instruction = """
Summarize the following review in one sentence:
The movie was visually stunning, with breathtaking cinematography and carefully crafted scenes that made every frame look like a piece of art.
The acting was also excellent, with the lead performers delivering emotional and convincing performances that kept the audience engaged throughout most of the film.
However, despite its strong visual and acting elements, the story felt too predictable.
Many plot twists could be easily anticipated, and the narrative followed a familiar formula without offering much originality.
"""

inputs = tokenizer(instruction, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

instruction = """
Summarize the following review in one sentence:
The movie was visually stunning, with breathtaking cinematography and carefully crafted scenes that made every frame look like a piece of art.
The acting was also excellent, with the lead performers delivering emotional and convincing performances that kept the audience engaged throughout most of the film.
However, despite its strong visual and acting elements, the story felt too predictable.
Many plot twists could be easily anticipated, and the narrative followed a familiar formula without offering much originality.
"""

inputs = tokenizer(instruction, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

instruction = """
Summarize the following review in one sentence:
The movie was visually stunning, with breathtaking cinematography and carefully crafted scenes that made every frame look like a piece of art.
The acting was also excellent, with the lead performers delivering emotional and convincing performances that kept the audience engaged throughout most of the film.
However, despite its strong visual and acting elements, the story felt too predictable.
Many plot twists could be easily anticipated, and the narrative followed a familiar formula without offering much originality.
"""

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": instruction}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs,max_new_tokens=60)

result = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)

print(result)