import torch
import json
import sys
import os
import tokenizer

sys.path.append(".")

from model import TransformerDecoderLM

def load_tokenizer(path: str):
    tok = tokenizer.BPETokenizer.load_tokenizer(path=path)
    return tok

tokenizer = load_tokenizer("bpe_tokenizer.json")

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("models/best_model.pth", map_location=device)

config = checkpoint.get("config", {
    "vocab_size": tokenizer.vocab_size,
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 8,
    "max_seq_len": 256
})

model = TransformerDecoderLM(
    vocab_size=max(tokenizer.vocab.values()) + 1,  # 🔥 критично!
    d_model=config["d_model"],
    n_layers=8,
    n_heads=config["n_heads"],
    max_seq_len=config.get("block_size", 256),
    dropout=0.0  # dropout не нужен при генерации
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()  # режим оценки!

print("✅ Модель и токенизатор загружены.")


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
    device="cpu"
):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        if input_ids.size(1) > model.max_seq_len:
            input_ids = input_ids[:, -model.max_seq_len:]

        logits = model(input_ids)  # (1, T, vocab)
        next_token_logits = logits[:, -1, :] / temperature  

        if top_k is not None:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        if next_token.item() == tokenizer.special_tokens.get("<EOS>", -1):
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())

prompts = [
    "Once upon a time grandfather took a crap in a stroller",
    "There was a little dog named Max.",
    "In a faraway land,",
]

for prompt in prompts:
    print(f"\n{'='*50}")
    print(f"Prompt: {prompt}")
    print(f"Generated:\n{generate(model, tokenizer, prompt, max_new_tokens=120, temperature=0.9, top_k=50, device=device)}")