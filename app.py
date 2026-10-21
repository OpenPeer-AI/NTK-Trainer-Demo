from __future__ import annotations
import io
import json
import math
import tempfile
import time
import torch
import gradio as gr
import spaces
from transformers import AutoTokenizer, AutoModelForCausalLM
from analysis import token_surprisal, surprisal_html
from styles import STYLES, HELD_OUT, NEUTRAL

MODEL_REPO = "OpenPeerAI/My-Name-is-Riemann-Prerelease"

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
)
model.eval()
N_PARAMS = sum(p.numel() for p in model.parameters())


def _gpu():
    if model.device.type != "cuda":
        model.to("cuda")


@torch.no_grad()
def mean_nll(texts):
    total, n = 0.0, 0
    for t in texts:
        r = token_surprisal(model, tokenizer, t, max_length=256)
        total += r["nll"] * r["n_tokens"]
        n += r["n_tokens"]
    return total / max(1, n)


def gate_plot(tuner):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = tuner.controller.raw.detach().float().cpu().numpy()
    layers = tuner.controller.layer_indices.detach().cpu().tolist()
    counts = {}
    for l in layers:
        counts[l] = counts.get(l, 0) + 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].hist(raw, bins=40, color="#4c6ef5")
    axes[0].set_title("Learned log-gate values s (layer output x exp(s))")
    axes[1].bar(list(counts.keys()), list(counts.values()), color="#20c997")
    axes[1].set_title("Gates per decoder layer")
    axes[1].set_xlabel("layer")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf


@spaces.GPU(duration=120)
def fit(
    target_style, gates, steps, lr, max_log_gate, probe_text, progress=gr.Progress()
):
    from ntkmirror import ForwardFineTuner, Example
    from PIL import Image

    _gpu()
    train = [Example(prompt="", completion=t) for t in STYLES[target_style]]
    eval_sets = {**HELD_OUT, "Neutral prose": NEUTRAL}

    progress(0.05, desc="Scoring base model on every style")
    before = {k: mean_nll(v) for k, v in eval_sets.items()}
    probe_before = token_surprisal(model, tokenizer, probe_text, max_length=256)

    tuner = ForwardFineTuner(
        model, tokenizer, gates=int(gates), max_log_gate=float(max_log_gate)
    )
    progress(0.2, desc=f"Fitting gates on {len(train)} {target_style} sentences")
    t0 = time.perf_counter()
    tuner.fit(
        train,
        steps=int(steps),
        lr=float(lr),
        batch_size=8,
        max_length=128,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    progress(0.8, desc="Scoring with controller")
    tuner.controller.attach()
    after = {k: mean_nll(v) for k, v in eval_sets.items()}
    probe_after = token_surprisal(model, tokenizer, probe_text, max_length=256)
    tuner.save(
        tmp_path := tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name,
        metadata={"demo": "ntk-trainer-space", "style": target_style},
    )
    tuner.controller.remove()
    n_gates = int(tuner.controller.raw.numel())

    rows = []
    for k in eval_sets:
        d = after[k] - before[k]
        mark = " (target)" if k == target_style else ""
        rows.append(
            f"| {k}{mark} | {math.exp(before[k]):.1f} | {math.exp(after[k]):.1f} | {d:+.3f} | {'lower' if d < -0.005 else ('higher' if d > 0.005 else 'about the same')} |"
        )
    table = (
        "| Held-out style | Perplexity, base | Perplexity, with controller | Change in mean surprisal (nats) | Direction |\n|---|---|---|---|---|\n"
        + "\n".join(rows)
    )

    target_gain = before[target_style] - after[target_style]
    others = [before[k] - after[k] for k in eval_sets if k != target_style]
    selectivity = target_gain - (sum(others) / len(others))

    probe_html = (
        surprisal_html(probe_before, "Base weights:")
        + "<hr/>"
        + surprisal_html(probe_after, f"With controller fitted on {target_style}:")
        + f"<p>Mean surprisal on this text: {probe_before['nll']:.3f} to {probe_after['nll']:.3f} nats ({probe_after['nll']-probe_before['nll']:+.3f}).</p>"
    )
    summary = {
        "base_model": MODEL_REPO,
        "base_parameters": N_PARAMS,
        "trainable_gates": n_gates,
        "trainable_fraction": n_gates / N_PARAMS,
        "target_style": target_style,
        "train_sentences": len(train),
        "steps": int(steps),
        "fit_seconds": round(elapsed, 1),
        "heldout_nll_before": {k: round(v, 4) for k, v in before.items()},
        "heldout_nll_after": {k: round(v, 4) for k, v in after.items()},
        "target_gain_nats": round(target_gain, 4),
        "mean_other_gain_nats": round(sum(others) / len(others), 4),
        "selectivity_nats": round(selectivity, 4),
    }
    report = (
        f"**Base model:** {MODEL_REPO}, {N_PARAMS/1e9:.1f}B parameters, frozen throughout.\n\n"
        f"**Controller:** {n_gates:,} gates ({100*n_gates/N_PARAMS:.5f}% of the model), fitted on {len(train)} {target_style} sentences in {int(steps)} steps ({elapsed:.1f}s). The saved controller file is a few kilobytes.\n\n"
        f"**Result on held-out sentences the gates never saw:**\n\n{table}\n\n"
        f"The controller lowered surprisal on held-out **{target_style}** by **{target_gain:+.3f} nats** while moving the other styles by {sum(others)/len(others):+.3f} on average, a selectivity of **{selectivity:+.3f} nats**. "
        "A positive selectivity means the gates learned something specific to the target style rather than a general shift. "
        "This is the same mechanism that tuned the 13.1B Prerelease toward its question-answering data: the weights never change, the controller re-weights what the model already represents."
    )
    img = Image.open(gate_plot(tuner))
    return report, probe_html, img, json.dumps(summary, indent=2), tmp_path


DESCRIPTION = f"""
# NTK Trainer Demo

Live demonstration of the
[OpenPeer-AI-NTK-Trainer](https://huggingface.co/OpenPeerAI/OpenPeer-AI-NTK-Trainer):
adapting a frozen language model by fitting a small set of **signed log-gates** on its layer outputs,
with the fitting objective derived from neural tangent kernel theory, instead of training LoRA adapters or
full weights.

The base model here is the **{N_PARAMS/1e9:.1f}B My Name is Riemann Prerelease**, the model this trainer
was actually used on. Pick a target writing style; the space fits a controller on forty sentences in that
style, then scores held-out sentences from **every** style, before and after. A working controller lowers
the model's surprise on the target style and leaves the others alone. You also see, token by token, how the
gates changed the model's view of any text you paste, and you can download the controller.

Gates re-weight what the model already knows. They do not add facts or teach new output formats, so this
demo measures what they change rather than asking them to produce a specific answer.
"""

with gr.Blocks(title="NTK Trainer Demo") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=2):
            style = gr.Dropdown(
                choices=list(STYLES),
                value="Legal and formal",
                label="Target style to fit the controller on",
            )
            with gr.Accordion(
                "Training sentences for this style (read-only preview)", open=False
            ):
                preview = gr.Textbox(
                    value="\n".join(STYLES["Legal and formal"][:8]) + "\n...",
                    lines=9,
                    interactive=False,
                    show_label=False,
                )
            style.change(lambda s: "\n".join(STYLES[s][:8]) + "\n...", style, preview)
            probe = gr.Textbox(
                label="Probe text for the token map (any sentence)",
                value="The parties agree that any notice shall be delivered in writing to the registered address.",
                lines=2,
            )
        with gr.Column(scale=1):
            gates = gr.Slider(256, 8192, value=2048, step=256, label="Gates to fit")
            steps = gr.Slider(20, 300, value=120, step=10, label="Fitting steps")
            lr = gr.Slider(1e-3, 3e-2, value=1e-2, step=1e-3, label="Learning rate")
            max_log_gate = gr.Slider(
                0.02, 0.5, value=0.2, step=0.02, label="Max |log gate|"
            )
            run = gr.Button("Fit controller and measure", variant="primary")
    report = gr.Markdown()
    probe_out = gr.HTML(label="Token-level effect on the probe text")
    with gr.Row():
        plot = gr.Image(label="Learned gates", type="pil")
        with gr.Column():
            summary = gr.Code(label="Summary JSON", language="json")
            download = gr.File(label="Download controller (.pt)")
    run.click(
        fit,
        [style, gates, steps, lr, max_log_gate, probe],
        [report, probe_out, plot, summary, download],
    )
    gr.Markdown(
        "OpenPeer-AI-NTK-Trainer is released under MIT. Depends on [ntkmirror](https://github.com/leochlon/ntkmirror). "
        "Maintained by OpenPeer AI."
    )

demo.queue(max_size=8).launch()
