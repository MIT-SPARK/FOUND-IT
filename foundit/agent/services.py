import base64
from io import BytesIO
from typing import List, Tuple

from foundit.agent.config import AgentConfig
from foundit.agent.interfaces import TaskAgentInterface
from foundit.agent.models import RegionResult, TaskResponse
from foundit.agent.prompts import get_system_prompt
from foundit.agent.tools import create_default_tool_registry
from foundit.objects import Bbox, Objects
from foundit.solver import Solver


class TaskAgent(TaskAgentInterface):
    """LLM agent that decomposes long-horizon tasks into object queries."""

    def __init__(self, config: AgentConfig, objects_module: Objects, solver: Solver):
        super().__init__(config)
        self.visual_verification = config.visual_verification
        self._has_places = solver.places is not None
        self.registry = create_default_tool_registry(config, objects_module, solver)

    def run(self, task: str) -> Tuple[TaskResponse, List[Bbox], List[RegionResult]]:
        all_bboxes: List[Bbox] = []
        all_regions: List[RegionResult] = []
        consecutive_empty = 0

        messages = [
            {"role": "system", "content": get_system_prompt(self.visual_verification, self._has_places)},
            {"role": "user", "content": task},
        ]

        tools = self.registry.get_openai_tools()

        for iteration in range(self.config.max_iterations):
            print(f"\n[Agent] Iteration {iteration + 1}/{self.config.max_iterations}")

            response = self.client.responses.create(
                model=self.config.model_name,
                input=messages,
                tools=tools,
                tool_choice="auto",
            )

            had_tool_calls = False
            found_results_this_iteration = False

            # Pass 1: Append all response output items (keeps reasoning + fc items contiguous)
            pending_tool_calls = []
            for item in response.output:
                messages.append(item)
                if item.type == "function_call":
                    had_tool_calls = True
                    pending_tool_calls.append(item)

            # Pass 2: Execute tools and append outputs
            for item in pending_tool_calls:
                result = self.registry.call_tool(item.name, item.arguments)

                if result.bboxes or result.regions:
                    found_results_this_iteration = True
                all_bboxes.extend(result.bboxes)
                all_regions.extend(result.regions)

                messages.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result.output_json,
                })

                if self.visual_verification and result.images:
                    image_content = []
                    for crop in result.images:
                        buf = BytesIO()
                        crop.convert("RGB").save(buf, format="JPEG", quality=80)
                        b64 = base64.b64encode(buf.getvalue()).decode()
                        image_content.append({
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{b64}",
                        })
                    image_content.append({
                        "type": "input_text",
                        "text": "Above are the cropped detection images. Verify each detection is correct for the query.",
                    })
                    messages.append({"role": "user", "content": image_content})

            if not had_tool_calls:
                break

            # early stop after consecutive empty iterations
            if not found_results_this_iteration:
                consecutive_empty += 1
                if consecutive_empty >= self.config.max_consecutive_empty:
                    print(f"[Agent] Stopping early: {self.config.max_consecutive_empty} consecutive empty iterations.")
                    break
            else:
                consecutive_empty = 0

        # Parse structured final response
        task_response = self._parse_final_response(messages)

        return task_response, all_bboxes, all_regions

    def _parse_final_response(self, messages: List) -> TaskResponse:
        # Filter out orphaned function_call items that lack a corresponding output
        # (can happen if we hit max iterations mid-call)
        output_ids = {
            msg.get("call_id") for msg in messages
            if isinstance(msg, dict) and msg.get("type") == "function_call_output"
        }
        clean_messages = []
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "function_call":
                if msg.call_id not in output_ids:
                    continue  # orphaned
            clean_messages.append(msg)

        clean_messages.append({
            "role": "user",
            "content": "Now provide your final structured response with reasoning, the list of all objects you found, and the list of all regions you found.",
        })

        response = self.client.responses.parse(
            model=self.config.model_name,
            input=clean_messages,
            text_format=TaskResponse,
        )

        # Extract the parsed response from the output
        for item in response.output:
            if item.type == "message" and item.content:
                for block in item.content:
                    if hasattr(block, "parsed") and block.parsed is not None:
                        return block.parsed

        # Fallback if parsing fails
        return TaskResponse(
            reasoning="Agent completed but could not generate structured response.",
            objects=[],
        )
