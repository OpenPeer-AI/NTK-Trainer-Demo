---
title: NTK Trainer Demo
emoji: 🎛️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
models:
  - OpenPeerAI/OpenPeer-AI-NTK-Trainer
  - OpenPeerAI/OpenPeerLLM
short_description: Fit NTK gate controllers on a frozen LLM live on ZeroGPU
---

# NTK Trainer Demo

A live demonstration of the
[OpenPeer-AI-NTK-Trainer](https://huggingface.co/OpenPeerAI/OpenPeer-AI-NTK-Trainer):
fine-tuning a frozen causal language model by fitting a small set of signed
log-gates on its layer outputs, with the fitting objective derived from
neural tangent kernel theory, instead of training LoRA adapters or full
weights.

Paste tab-separated prompt and completion pairs, choose a gate budget and
step count, and the space loads
[OpenPeerLLM](https://huggingface.co/OpenPeerAI/OpenPeerLLM) on ZeroGPU,
scores a held-out slice, selects and fits the gates with
[ntkmirror](https://github.com/leochlon/ntkmirror), scores again, shows
greedy continuations before and after, plots the learned gates, and offers
the controller file for download.

The same method was used at every rung of the
[My Name is Riemann](https://huggingface.co/OpenPeerAI/My-Name-is-Riemann-Prerelease)
growth ladder.

Maintained by OpenPeer AI. The trainer is released under MIT.
