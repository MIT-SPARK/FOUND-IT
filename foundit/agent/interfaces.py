from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from openai import OpenAI

from foundit.agent.config import AgentConfig


class Tool(ABC):
    """Abstract base class for agent tools.

    Tools return a dict from execute(). Special sentinel keys:
      - "_images": list of PIL images for visual context injection
      - "_bboxes": list of Bbox objects for the caller to collect
      - "_regions": list of RegionResult objects for the caller to collect
    The service layer strips sentinels before serializing the rest as JSON.
    """

    def __init__(self):
        self.name: str = ""
        self.description: str = ""
        self.signature: Dict = {}
        self._build_signature()

    def _build_signature(self) -> None:
        self.signature = {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema(),
            "strict": True,
        }

    @abstractmethod
    def _get_parameters_schema(self) -> Dict:
        """Return JSON schema for tool parameters."""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> Dict:
        """Execute the tool with the given parameters."""
        ...

    def set_shared_state(self, **kwargs) -> None:
        """Receive shared state from the registry. No-op by default."""
        pass


class TaskAgentInterface(ABC):
    """Abstract base class for the task-planning agent."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = OpenAI()

    @abstractmethod
    def run(self, task: str) -> Tuple:
        """Run the agent on a long-horizon task description.

        Returns:
            Tuple of (TaskResponse, List[Bbox], List[RegionResult])
        """
        ...
