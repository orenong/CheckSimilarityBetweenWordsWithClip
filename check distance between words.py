import torch
from torch.nn import CosineSimilarity
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel

cossim = CosineSimilarity(dim=0, eps=1e-6)


def dist(v1, v2):
    return cossim(v1, v2)


torch_device = "cuda" if torch.cuda.is_available() else "cpu"

models = [
    'openai/clip-vit-base-patch16',
    'openai/clip-vit-base-patch32',
    'openai/clip-vit-large-patch14',
]

model_id = models[1]

tokenizer = CLIPTokenizer.from_pretrained(model_id)
text_encoder = CLIPTextModel.from_pretrained(model_id).to(torch_device)
model = CLIPModel.from_pretrained(model_id).to(torch_device)

prompts = ['Audi', 'Mazda', 'Ford', 'Toyota', 'Nissan', 'BMW', 'subaru', 'Car', 'Cat', 'Audio', 'putin', 'donald trump', 'vladimir putin', 'russia?', 'apple', 'green animal', 'automation', 'Auto', 'auto!??!!?', 'Automation', 'Gnusie Shaboozie', 'oren', 'russia', 'red fruit', 'giant iguana'

]

words_to_check = ["Auto","Gnusie Shaboozie","I'm testing something"] # <<--------------------------------------------------------change here

for word in words_to_check:
    if not word in prompts:
        prompts.append(word)

text_inputs = tokenizer(
    prompts,
    padding="max_length",
    return_tensors="pt",
).to(torch_device)
text_features = model.get_text_features(**text_inputs)
text_embeddings = torch.flatten(text_encoder(text_inputs.input_ids.to(torch_device))['last_hidden_state'], 1, -1)



print("\n\nusing text_embeddings")
for label1 in words_to_check:
    results = {}
    for i2, label2 in enumerate(prompts):
        i1 = prompts.index(label1)
        results[f"{label1} <-> {label2}"] = dist(text_embeddings[i1], text_embeddings[i2])
    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1],reverse=True)}

    for key, value in sorted_results.items():
        formatted_value = "{:.4f}".format(value)
        print(f"{key} {formatted_value}")


