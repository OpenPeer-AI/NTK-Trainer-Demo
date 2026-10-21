from __future__ import annotations
import html
import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def token_surprisal(model, tokenizer, text: str, max_length: int = 1024):
    ids = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).input_ids.to(model.device)
    logits = model(input_ids=ids).logits[0, :-1].float()
    targets = ids[0, 1:]
    logprobs = F.log_softmax(logits, dim=-1)
    tok_lp = logprobs.gather(1, targets[:, None]).squeeze(1)
    ranks = (logprobs > tok_lp[:, None]).sum(dim=1) + 1
    tokens = [tokenizer.decode([t]) for t in ids[0].tolist()]
    nll = -tok_lp.mean().item()
    return {
        "tokens": tokens,
        "surprisal": [-x for x in tok_lp.tolist()],
        "rank": ranks.tolist(),
        "nll": nll,
        "ppl": math.exp(nll),
        "n_tokens": int(targets.numel()),
    }


def surprisal_html(result: dict, title: str = "") -> str:
    toks, surp = result["tokens"], result["surprisal"]
    cells = [f"<span style='color:#888'>{html.escape(toks[0])}</span>"]
    for t, s in zip(toks[1:], surp):
        p = min(1.0, s / 12.0)
        r, g = int(255 * p), int(200 * (1 - p))
        cells.append(
            f"<span title='surprisal {s:.2f} nats' style='background:rgba({r},{g},80,0.35);padding:1px 2px;border-radius:3px'>{html.escape(t)}</span>"
        )
    head = (
        f"<div style='margin-bottom:6px'><b>{html.escape(title)}</b> perplexity <b>{result['ppl']:.1f}</b> over {result['n_tokens']} tokens (mean surprisal {result['nll']:.2f} nats)</div>"
        if title
        else ""
    )
    return (
        head
        + "<div style='font-family:ui-monospace,monospace;line-height:1.9;white-space:pre-wrap;word-break:break-word'>"
        + "".join(cells)
        + "</div><div style='color:#777;font-size:0.85em;margin-top:4px'>green = expected, red = surprising; hover a token for its surprisal</div>"
    )


@torch.no_grad()
def greedy_continue(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    repetition_penalty: float = 1.2,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_context: int = 1024,
):
    ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_context - max_new_tokens,
    ).input_ids.to(model.device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-3),
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)


def contrast_rows(model, tokenizer, samples: list[tuple[str, str]]):
    rows = []
    for label, text in samples:
        r = token_surprisal(model, tokenizer, text)
        rows.append([label, round(r["ppl"], 1), round(r["nll"], 3), text])
    return rows


def md_table(rows, header=("Text", "Perplexity", "Mean surprisal (nats)")):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append(f"| {r[0]} | {r[1]} | {r[2]} |")
    return "\n".join(out)


WIKI_FEWSHOT = (
    "Paris is the capital and most populous city of France. It is located on the Seine river in the north of the country.\n\n"
    "Tokyo is the capital of Japan and one of the most populous cities in the world. It lies on the eastern coast of Honshu.\n\n"
)

QA_FEWSHOT = (
    "Q: What is the capital of France?\nA: Paris.\n\n"
    "Q: Which river flows through Cairo?\nA: The Nile.\n\n"
    "Q: What is the largest planet in the Solar System?\nA: Jupiter.\n\n"
)

CONTRAST_SET = [
    (
        "Fluent English (encyclopedic)",
        "The Amazon River in South America is the largest river by discharge volume of water in the world.",
    ),
    (
        "Fluent English (narrative)",
        "She closed the book, turned off the lamp, and listened to the rain against the window.",
    ),
    (
        "Grammatical but false",
        "The Amazon River is located in northern Canada and flows into the Arctic Ocean near Greenland.",
    ),
    (
        "Word salad (same words shuffled)",
        "world the in water of volume discharge by river largest the is America South in River Amazon The",
    ),
    ("Random words", "purple elephant calculus whisper nine umbrella fjord lantern"),
    ("Python", "def add(a, b):\n    return a + b"),
]
