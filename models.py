from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid

class ScoreGrid(BaseModel):
    cryptography: int = Field(default=0)
    history: int = Field(default=0)
    geography: int = Field(default=0)
    logic: int = Field(default=0)

class Critique(BaseModel):
    weaknesses: List[str] = Field(default_factory=list, description="List of identified weaknesses in the idea.")
    score_grid: Optional[ScoreGrid] = None
    feedback: str = Field(..., description="A detailed textual critique.")

class Idea(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., description="The reasoning path or hypothesis.")
    parent_id: Optional[str] = Field(default=None, description="ID of the parent idea this evolved from.")
    total_score: float = Field(default=0.0)
    abstracted_facts: Optional[Dict[str, str]] = None
    generation_depth: int = Field(default=0)
    critique: Optional[Critique] = None


class RiddleState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str = Field(..., description="Original riddle text.")
    image_descriptions: List[str] = Field(default_factory=list, description="Visual hints extracted by Gemini.")
    ideas: Dict[str, Idea] = Field(default_factory=dict, description="All ideas generated, keyed by ID.")
    current_generation: int = Field(default=0)
    top_idea_id: Optional[str] = Field(default=None, description="The current best idea to branch from.")

    def get_generation_ideas(self, depth: int) -> List[Idea]:
        return [idea for idea in self.ideas.values() if idea.generation_depth == depth]
