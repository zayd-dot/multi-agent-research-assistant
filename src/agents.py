"""Agent definitions for the multi-agent research system.

Each agent is a specialized LLM with a focused system prompt and
access to specific tools. Agents communicate through shared state.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List

from configs.config import LLMConfig
from src.tools import web_search, summarize_text, fact_check


def _build_llm(cfg: LLMConfig = LLMConfig()) -> ChatOpenAI:
    return ChatOpenAI(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens)


RESEARCHER_PROMPT = """You are a Research Agent. Your job is to:
1. Break down the user's research topic into specific search queries
2. Use the web_search tool to find relevant, recent information
3. Collect and organize raw findings

Always search for multiple angles on the topic. Focus on authoritative sources.
Return your findings as structured notes with source URLs."""


SUMMARIZER_PROMPT = """You are a Summarization Agent. Your job is to:
1. Take raw research findings and distill them into clear, organized summaries
2. Use the summarize_text tool for long passages
3. Identify key themes, consensus views, and notable disagreements
4. Organize information by subtopic

Produce well-structured summaries that preserve important details and nuance."""


FACT_CHECKER_PROMPT = """You are a Fact-Checking Agent. Your job is to:
1. Review claims made in the research summary
2. Use the fact_check tool to verify key factual claims
3. Flag any claims that are disputed or unverified
4. Note the confidence level of each verification

Be thorough but focus on the most important or surprising claims first."""


WRITER_PROMPT = """You are a Report Writer Agent. Your job is to:
1. Take verified research summaries and produce a polished final report
2. Structure the report with clear sections: Overview, Key Findings, Analysis, Conclusion
3. Include source citations throughout
4. Flag any claims that were disputed during fact-checking
5. Write in a clear, professional tone

The report should be comprehensive yet concise, suitable for a knowledgeable audience."""


def get_researcher(cfg: LLMConfig = LLMConfig()):
    """Create the Research Agent with web search capability."""
    llm = _build_llm(cfg)
    return llm.bind_tools([web_search]), RESEARCHER_PROMPT


def get_summarizer(cfg: LLMConfig = LLMConfig()):
    """Create the Summarization Agent."""
    llm = _build_llm(cfg)
    return llm.bind_tools([summarize_text]), SUMMARIZER_PROMPT


def get_fact_checker(cfg: LLMConfig = LLMConfig()):
    """Create the Fact-Checking Agent."""
    llm = _build_llm(cfg)
    return llm.bind_tools([fact_check]), FACT_CHECKER_PROMPT


def get_writer(cfg: LLMConfig = LLMConfig()):
    """Create the Report Writer Agent."""
    llm = _build_llm(cfg)
    return llm, WRITER_PROMPT
