from typing import List

from pydantic import BaseModel


class BboxResult(BaseModel):
    label: str
    center: List[float]
    extent: List[float]


class RegionResult(BaseModel):
    label: str
    centroid: List[float]
    extent: List[float]
    num_nodes: int
    confidence: float


class TaskResponse(BaseModel):
    reasoning: str
    objects: List[BboxResult]
    regions: List[RegionResult] = []
