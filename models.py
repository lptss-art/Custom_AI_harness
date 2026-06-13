from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid

class ScoreGrid(BaseModel):
    advancement: int = Field(default=0, description="Does it seem to advance the riddle?")
    coherence: int = Field(default=0, description="Does it seem coherent?")
    plausibility: int = Field(default=0, description="Does it seem plausible?")

class Critique(BaseModel):
    weaknesses: List[str] = Field(default_factory=list, description="List of identified weaknesses in the idea.")
    score_grid: Optional[ScoreGrid] = None
    feedback: str = Field(..., description="A detailed textual critique.")

class PisteResolution(BaseModel):
    id_piste: str = Field(default_factory=lambda: f"PISTE-{str(uuid.uuid4())[:8].upper()}", description="Identifiant unique")
    statut: str = Field(default="En attente", description="Statut : En attente / Active / Validée / Fausse Piste / Bloquée par Parent")
    noeud_graphe_origine: str = Field(default="Racine", description="Énigme ou Pit Stop associé")
    pistes_parentes: List[str] = Field(default_factory=list, description="IDs des pistes dont elle découle")
    pistes_enfants: List[str] = Field(default_factory=list, description="IDs des sous-pistes générées par celle-ci")
    hypothese_de_depart: str = Field(..., description="The reasoning path or hypothesis.")
    protocole_de_test: str = Field(default="Aucun", description="Script Python ou méthode de vérification")
    resultat_du_test: Optional[str] = None
    output_simple: Optional[str] = Field(default=None, description="Résultat réutilisable de la piste (ex: texte décrypté).")
    probleme_rencontre: Optional[str] = None
    solution_proposee: Optional[str] = None
    analyse_avocat_du_diable: Optional[Critique] = None
    score_elo: float = Field(default=0.0)
    raison_blocage: Optional[str] = None
    abstracted_facts: Optional[Dict[str, str]] = None
    generation_depth: int = Field(default=0)

class RiddleState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str = Field(..., description="Original riddle text.")
    image_descriptions: List[str] = Field(default_factory=list, description="Visual hints extracted by Gemini.")
    pistes: Dict[str, PisteResolution] = Field(default_factory=dict, description="All pistes generated, keyed by ID.")
    current_generation: int = Field(default=0)
    top_piste_id: Optional[str] = Field(default=None, description="The current best piste to branch from.")

    def get_generation_pistes(self, depth: int) -> List[PisteResolution]:
        return [piste for piste in self.pistes.values() if piste.generation_depth == depth]
