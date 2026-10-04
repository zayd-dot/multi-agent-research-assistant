"""Tools for research agents. All free, no API keys."""

from duckduckgo_search import DDGS
from typing import List, Dict, Any
import torch


def web_search(query, max_results=5):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [{"title": r.get("title",""), "url": r.get("href",""), "content": r.get("body","")} for r in results]
    except Exception as e:
        return [{"title": "Error", "url": "", "content": f"Search failed: {e}"}]


def fact_check(claim, model, tokenizer, device):
    results = web_search(claim, max_results=3)
    evidence = [{"source": r["url"], "excerpt": r["content"][:300]} for r in results if r.get("content")]
    evidence_text = "\n".join(f"- {e['excerpt']}" for e in evidence)

    prompt = (
        f"Based on the evidence, is this claim SUPPORTED, DISPUTED, or UNVERIFIED?\n\n"
        f"Claim: {claim}\n\nEvidence:\n{evidence_text}\n\nVerdict and reasoning:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    verdict = "UNVERIFIED"
    for v in ["SUPPORTED", "DISPUTED"]:
        if v.lower() in response.lower():
            verdict = v
            break

    return {"verdict": verdict, "reasoning": response, "sources": evidence, "claim": claim}
