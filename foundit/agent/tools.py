import json
import numpy as np
from typing import Dict, List, Optional

from foundit.agent.interfaces import Tool
from foundit.agent.models import RegionResult
from foundit.objects import Bbox, Detection, Objects
from foundit.places.clustering.query import create_query_clusterer, query_region
from foundit.places.config import QueryClusterConfig
from foundit.slam_utils import overlay_contour
from foundit.solver import Solver


class ToolCallResult:
    """Separates LLM-visible JSON from sentinel data."""

    def __init__(self, output_json: str, bboxes: list = None, images: list = None, regions: list = None):
        self.output_json = output_json
        self.bboxes = bboxes or []
        self.images = images or []
        self.regions = regions or []


class FindObjectTool(Tool):
    """Tool that searches for a 3D object in the reconstructed scene. If visual_verification is enabled, also verifies cropped predictions via LLM."""

    def __init__(self, objects_module: Objects, solver: Solver, visual_verification: bool = False):
        self._objects = objects_module
        self._solver = solver
        self._visual_verification = visual_verification
        super().__init__()
        self.name = "find_object"
        self.description = (
            "Search for a specific object in the 3D scene by text query. "
            "Returns 3D bounding boxes for all instances found. "
            "DON'T use complex object names, keep it simple."
        )
        self._build_signature()

    def _get_parameters_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The object to search for (e.g., 'chair', 'table', 'fork')",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    @staticmethod
    def _create_prompt_crop(image, mask, padding=0.1, min_pad=10):
        """Crop and contour object region in image. Padding added around the bounding box to improve LLM understanding."""

        contoured = overlay_contour(image, mask)

        mask_2d = mask.squeeze()
        rows = np.where(mask_2d.any(axis=1))[0]
        cols = np.where(mask_2d.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return contoured

        r_min, r_max = rows[0], rows[-1]
        c_min, c_max = cols[0], cols[-1]
        pad_r = max(int(padding * (r_max - r_min)), min_pad)
        pad_c = max(int(padding * (c_max - c_min)), min_pad)
        r_min = max(0, r_min - pad_r)
        r_max = min(image.height, r_max + pad_r)
        c_min = max(0, c_min - pad_c)
        c_max = min(image.width, c_max + pad_c)

        return contoured.crop((c_min, r_min, c_max, r_max))

    def execute(self, **kwargs) -> Dict:
        query = kwargs["query"]
        print(f"\n[Agent] Searching for: '{query}'")

        detected_objects = self._objects.get_3D_object(query, self._solver)
        bboxes = [obj.bbox for obj in detected_objects]

        result: Dict = {"_bboxes": bboxes}

        # Generate crops with contour overlays for visual verification
        if self._visual_verification:
            crops = []
            for obj in detected_objects:
                if obj.mask is not None and obj.source_image is not None:
                    crops.append(self._create_prompt_crop(obj.source_image, obj.mask))

            if crops:
                result["_images"] = crops

        if not bboxes:
            result["message"] = f"No objects found for query: '{query}'"
        else:
            result["message"] = f"Found {len(bboxes)} object(s) for query: '{query}'"
            result["objects"] = [
                {"label": b.label or query, "center": [float(c) for c in b.center], "extent": [float(e) for e in b.extent]}
                for b in bboxes
            ]
        return result


class FindObjectsTool(Tool):
    """Tool that searches for multiple 3D objects simultaneously with batched inference."""

    def __init__(self, objects_module: Objects, solver: Solver, visual_verification: bool = False):
        self._objects = objects_module
        self._solver = solver
        self._visual_verification = visual_verification
        super().__init__()
        self.name = "find_objects"
        self.description = (
            "Search for multiple objects simultaneously in the 3D scene. "
            "More efficient than calling find_object repeatedly. "
            "Use when the task involves finding 2 or more different objects. "
            "DON'T use complex object names, keep them simple."
        )
        self._build_signature()

    def _get_parameters_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of objects to search for (e.g., ['chair', 'table', 'cup'])",
                }
            },
            "required": ["queries"],
            "additionalProperties": False,
        }

    def execute(self, **kwargs) -> Dict:
        queries = kwargs["queries"]
        print(f"\n[Agent] Searching for {len(queries)} objects: {queries}")

        all_detections = self._objects.get_3D_objects(queries, self._solver)
        all_bboxes = []
        result: Dict = {}

        objects_list = []
        for query in queries:
            detections = all_detections.get(query, [])
            bboxes = [obj.bbox for obj in detections]
            all_bboxes.extend(bboxes)

            if not bboxes:
                objects_list.append({"query": query, "count": 0, "message": f"No objects found for '{query}'"})
            else:
                objects_list.append({
                    "query": query,
                    "count": len(bboxes),
                    "instances": [
                        {"label": b.label or query, "center": [float(c) for c in b.center],
                         "extent": [float(e) for e in b.extent]}
                        for b in bboxes
                    ],
                })

        result["_bboxes"] = all_bboxes

        # Generate crops with contour overlays for visual verification
        if self._visual_verification:
            crops = []
            for query in queries:
                for obj in all_detections.get(query, []):
                    if obj.mask is not None and obj.source_image is not None:
                        crops.append(FindObjectTool._create_prompt_crop(obj.source_image, obj.mask))
            if crops:
                result["_images"] = crops

        total_found = sum(len(all_detections.get(q, [])) for q in queries)
        result["message"] = f"Found {total_found} object(s) across {len(queries)} queries."
        result["results"] = objects_list
        return result


class FindRegionTool(Tool):
    """Tool that finds spatial regions in the navigable place graph by text query."""

    def __init__(self, objects_module: Objects, solver: Solver) -> None:
        self._solver = solver
        self._clusterer = create_query_clusterer(
            config=QueryClusterConfig(),
            clip_model=objects_module.clip_model,
            clip_tokenizer=objects_module.clip_tokenizer,
        )
        super().__init__()
        self.name = "find_region"
        self.description = (
            "Search for a spatial region or area in the navigable map by text query. "
            "Returns region centroid, spatial extent, and confidence. "
            "Use for area/room queries."
        )
        self._build_signature()

    def _get_parameters_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The region or area to search for",
                },
                "multi_region": {
                    "type": "boolean",
                    "description": (
                        "Whether to return multiple matching regions or only the single best match. "
                        "Set true for plural or multi-instance queries (e.g., 'bathrooms') or unbounded areas (e.g., 'street'), "
                        "false for singular queries (e.g., 'bathroom')."
                    ),
                },
            },
            "required": ["query", "multi_region"],
            "additionalProperties": False,
        }

    def execute(self, **kwargs) -> Dict:
        query = kwargs["query"]
        multi_region: bool = kwargs["multi_region"]
        print(f"\n[Agent] Searching for region: '{query}' (multi={multi_region})")

        assert self._solver.places is not None, (
            "Places module not initialized. Run with --places to enable region queries."
        )

        output = query_region(self._clusterer, self._solver.places.graph, query, multi_region)

        if output is None:
            return {"message": f"No regions found for query: '{query}'"}

        self._solver.viewer.visualize_clusters(output.vis_result, self._solver.places)

        region_models = [
            RegionResult(
                label=r.label, centroid=r.centroid, extent=r.extent,
                num_nodes=r.num_nodes, confidence=r.confidence,
            )
            for r in output.regions
        ]
        regions = [
            {"centroid": r.centroid, "extent": r.extent,
             "num_nodes": r.num_nodes, "confidence": round(r.confidence, 4)}
            for r in output.regions
        ]

        return {
            "message": f"Found {len(regions)} region(s) for query: '{query}'",
            "regions": regions,
            "_regions": region_models,
        }


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def set_shared_state(self, **kwargs) -> None:
        for tool in self._tools.values():
            tool.set_shared_state(**kwargs)

    def get_openai_tools(self) -> List[Dict]:
        return [tool.signature for tool in self._tools.values()]

    def call_tool(self, name: str, arguments: str) -> ToolCallResult:
        if name not in self._tools:
            return ToolCallResult(output_json=json.dumps({"error": f"Unknown tool: {name}"}))
        kwargs = json.loads(arguments)
        raw = self._tools[name].execute(**kwargs)
        bboxes = raw.pop("_bboxes", [])
        images = raw.pop("_images", [])
        regions = raw.pop("_regions", [])
        return ToolCallResult(output_json=json.dumps(raw), bboxes=bboxes, images=images, regions=regions)

    def get_tool(self, name: str) -> Tool:
        return self._tools[name]


def create_default_tool_registry(config, objects_module: Objects, solver: Solver) -> ToolRegistry:
    available_tools = {
        "find_object": lambda: FindObjectTool(objects_module, solver, visual_verification=config.visual_verification),
        "find_objects": lambda: FindObjectsTool(objects_module, solver, visual_verification=config.visual_verification),
    }
    if solver.places is not None:
        available_tools["find_region"] = lambda: FindRegionTool(objects_module, solver)

    registry = ToolRegistry()
    names = config.tools if config.tools is not None else list(available_tools.keys())
    for name in names:
        if name not in available_tools:
            raise ValueError(f"Unknown tool: '{name}'. Available: {list(available_tools.keys())}")
        registry.register(available_tools[name]())
    return registry
