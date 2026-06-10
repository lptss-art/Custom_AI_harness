from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid

class Critique(BaseModel):
    weaknesses: List[str] = Field(default_factory=list, description="List of identified weaknesses in the idea.")
    alignment_score: float = Field(default=0.0, description="Score from 0 to 1 indicating how well the idea aligns with known facts.")
    feedback: str = Field(..., description="A detailed textual critique.")

class Idea(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., description="The reasoning path or hypothesis.")
    parent_id: Optional[str] = Field(default=None, description="ID of the parent idea this evolved from.")
    elo_rating: float = Field(default=1200.0)
    generation_depth: int = Field(default=0)
    critique: Optional[Critique] = None

    def expected_score(self, other_rating: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((other_rating - self.elo_rating) / 400.0))

    def update_elo(self, actual_score: float, k_factor: float = 32.0):
        expected = self.expected_score(self.elo_rating) # Note: Needs opponent rating passed correctly, handled in Engine
        pass # Actual ELO update is handled in the engine during matchups

class RiddleState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str = Field(..., description="Original riddle text.")
    image_descriptions: List[str] = Field(default_factory=list, description="Visual hints extracted by Gemini.")
    ideas: Dict[str, Idea] = Field(default_factory=dict, description="All ideas generated, keyed by ID.")
    current_generation: int = Field(default=0)
    top_idea_id: Optional[str] = Field(default=None, description="The current best idea to branch from.")

    def get_generation_ideas(self, depth: int) -> List[Idea]:
        return [idea for idea in self.ideas.values() if idea.generation_depth == depth]
