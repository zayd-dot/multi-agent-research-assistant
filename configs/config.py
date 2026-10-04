"""Configuration for Multi-Agent Research Assistant (fully local, no API keys)."""

from dataclasses import dataclass


@dataclass
class LLMConfig:
    model_name: str = "google/flan-t5-large"
    max_new_tokens: int = 512
    temperature: float = 0.2


@dataclass
class AgentConfig:
    max_research_iterations: int = 2
    max_claims_to_check: int = 3
