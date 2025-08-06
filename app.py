import os
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# 📦 App version
APP_VERSION = "v1.1"

# Pull your HF token from env; set HF_TOKEN in GitHub Secrets or locally
HF_TOKEN    = os.getenv("HF_TOKEN", None)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID    = "gpt2"  # replace with any open model ID

def load_model_and_tokenizer():
    """Load model & tokenizer (cached) with optional HF_TOKEN auth."""
    kwargs = {"use_auth_token": HF_TOKEN} if HF_TOKEN else {}
    tok = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs).to(DEVICE)
    return tok, mdl

def generate(prompt, max_new_tokens, temperature, top_k):
    tok, mdl = load_model_and_tokenizer()
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    out    = mdl.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tok.eos_token_id
    )
    return tok.decode(out[0], skip_special_tokens=True)

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown(f"## 🧩 FFT Compression Playground ({APP_VERSION})")
        prompt = gr.Textbox(lines=3, placeholder="Enter prompt here...", label="Prompt")
        with gr.Row():
            max_toks   = gr.Slider(10, 200, value=100, step=1, label="Max New Tokens")
            temp       = gr.Slider(0.2, 1.5, value=0.9, step=0.1, label="Temperature")
            topk       = gr.Slider(10, 200, value=50, step=1, label="Top-k Sampling")
        output = gr.Textbox(lines=10, label="Generated Output")
        btn    = gr.Button("Generate")
        btn.click(fn=generate,
                  inputs=[prompt, max_toks, temp, topk],
                  outputs=output)
    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))

