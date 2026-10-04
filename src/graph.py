"""Multi-agent research pipeline — fully local, no API keys."""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, Any

from configs.config import LLMConfig, AgentConfig
from src.tools import web_search, fact_check


def _load_llm(cfg: LLMConfig = LLMConfig()):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {cfg.model_name} on {device.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name).to(device)
    print("Model loaded.")
    return model, tokenizer, device


def _generate(model, tokenizer, device, prompt, max_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def research_agent(state, model, tokenizer, device):
    topic = state["topic"]
    iteration = state.get("iteration", 0)

    query_prompt = f"Generate 3 search queries to research this topic: {topic}. List them numbered 1-3."
    queries_text = _generate(model, tokenizer, device, query_prompt, 128)

    queries = [line.strip().lstrip("0123456789.-) ") for line in queries_text.split("\n") if line.strip()]
    if len(queries) < 2:
        queries = [topic, f"{topic} latest developments", f"{topic} research"]

    all_results = []
    for q in queries[:3]:
        print(f"    🔎 Searching: {q}")
        for r in web_search(q, max_results=3):
            if r.get("content"):
                all_results.append(f"**{r['title']}** ({r['url']})\n{r['content']}")

    raw = "\n\n".join(all_results[:6])
    synthesis_prompt = f"Organize these search results about '{topic}' into key findings:\n\n{raw[:2500]}"
    synthesis = _generate(model, tokenizer, device, synthesis_prompt, 512)

    notes = state.get("research_notes", [])
    notes.append(synthesis)
    return {**state, "research_notes": notes, "iteration": iteration + 1, "status": "researched"}


def summarize_agent(state, model, tokenizer, device):
    all_notes = "\n\n".join(state.get("research_notes", []))
    prompt = (
        f"Create a comprehensive summary of this research on '{state['topic']}'."
        f" Organize by theme. Highlight key facts:\n\n{all_notes[:2500]}"
    )
    summary = _generate(model, tokenizer, device, prompt, 512)
    return {**state, "summary": summary, "status": "summarized"}


def fact_check_agent(state, model, tokenizer, device, cfg=AgentConfig()):
    claims_prompt = f"List the 3 most important factual claims from this text:\n\n{state.get('summary', '')[:1500]}"
    claims_text = _generate(model, tokenizer, device, claims_prompt, 256)
    claims = [line.strip().lstrip("0123456789.-) ") for line in claims_text.split("\n")
              if line.strip() and len(line.strip()) > 10]

    results = []
    for claim in claims[:cfg.max_claims_to_check]:
        print(f"    ✅ Checking: {claim[:80]}...")
        result = fact_check(claim, model, tokenizer, device)
        results.append(result)

    return {**state, "fact_check_results": results, "status": "fact_checked"}


def writer_agent(state, model, tokenizer, device):
    fc_text = ""
    for fc in state.get("fact_check_results", []):
        fc_text += f"- [{fc['verdict']}] {fc['claim']}\n"

    prompt = (
        f"Write a research report on '{state['topic']}' with sections: "
        f"Executive Summary, Key Findings, Analysis, Fact-Check Results, Conclusion.\n\n"
        f"Research Summary:\n{state.get('summary', '')[:1500]}\n\n"
        f"Fact-Check Results:\n{fc_text}"
    )
    report = _generate(model, tokenizer, device, prompt, 512)
    return {**state, "final_report": report, "status": "complete"}


def run_pipeline(topic, cfg_llm=LLMConfig(), cfg_agent=AgentConfig()):
    model, tokenizer, device = _load_llm(cfg_llm)

    state = {
        "topic": topic, "research_notes": [], "summary": "",
        "fact_check_results": [], "final_report": "", "status": "started", "iteration": 0,
    }

    for i in range(cfg_agent.max_research_iterations):
        print(f"\n🔍 Research Agent (iteration {i+1}/{cfg_agent.max_research_iterations})")
        state = research_agent(state, model, tokenizer, device)

    print(f"\n📝 Summarization Agent")
    state = summarize_agent(state, model, tokenizer, device)
    print(f"   Summary: {len(state['summary'])} chars")

    print(f"\n✅ Fact-Check Agent")
    state = fact_check_agent(state, model, tokenizer, device, cfg_agent)
    for fc in state["fact_check_results"]:
        print(f"   [{fc['verdict']}] {fc['claim'][:60]}...")

    print(f"\n📄 Writer Agent")
    state = writer_agent(state, model, tokenizer, device)
    print(f"   Report: {len(state['final_report'])} chars")

    return state
