from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AgentConfig:
    model_name: str = "gpt-5-mini"
    max_iterations: int = 15
    max_consecutive_empty: int = 2
    visual_verification: bool = False
    tools: Optional[List[str]] = None # None = use all registered tools
