import pandas as pd
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM



path = "dataset_builder/datasets/train/svamp-train.json"

dataset = pd.read_json(path)

for i, body in enumerate(dataset['Body']):
    if not body.rstrip().endswith('.'):
        dataset.at[i, 'Body'] = body.rstrip() + '.'

questions = dataset.Body + ' ' + dataset.Question


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = tokenizer(questions[0], return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

'''for q in questions:
    output = llm(q, max_tokens=128, stop=["<|end|>", "<|user|>"])
    answers.append(output)
    print(output["choices"][0]["text"])'''