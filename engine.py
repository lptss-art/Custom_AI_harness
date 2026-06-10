import asyncio
import itertools
from typing import List, Tuple
from config import deepseek_client, gemini_client, K_FACTOR, BASE_ELO
from models import Idea, Critique, RiddleState

PROPOSER_PROMPT = """You are the Proposer Agent in a multi-agent system solving a treasure hunt riddle.
Your goal is to generate a unique, highly logical, and concrete hypothesis or reasoning path.
Do not hallucinate. Base your reasoning STRICTLY on the given context, visual descriptions, and the parent reasoning path (if any).
Provide only the reasoning text. Be direct and concise."""

CRITIC_PROMPT = """You are the Critic Agent. Your job is to rigorously cross-examine the given hypothesis.
Look for logical flaws, historical inaccuracies, cryptographical errors, and inconsistencies with the provided visual hints.
You must return a JSON-like structure (or text parseable as such) outlining the weaknesses, an alignment score (0.0 to 1.0), and a detailed feedback string."""

JUDGE_PROMPT = """You are the Elo Tournament Judge.
You are evaluating two competing ideas to solve a riddle.
Context: {context}

Idea A: {idea_a}
Critique of Idea A: {critique_a}

Idea B: {idea_b}
Critique of Idea B: {critique_b}

Compare the two ideas based on logic, consistency with clues, the rigor of their critiques, and likelihood of being the correct step.
Output ONLY 'A' if Idea A is better, or 'B' if Idea B is better."""


class NexusEngine:
    def __init__(self, state: RiddleState):
        self.state = state

    async def analyze_image_with_gemini(self, image_data: bytes, mime_type: str) -> str:
        """Uses Gemini to extract visual hints from an image."""
        try:
            # Assuming image_data is raw bytes and mime_type is provided (e.g., 'image/jpeg')
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    "Extract all hidden symbols, text, and visual clues relevant to a treasure hunt riddle from this image.",
                    {'mime_type': mime_type, 'data': image_data}
                ]
            )
            return response.text
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    async def _propose_single_path(self, context_str: str, parent_id: str = None, depth: int = 0) -> Idea:
        """Asks DeepSeek to generate one hypothesis."""
        try:
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": PROPOSER_PROMPT},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nPropose a hypothesis:"}
                ],
                temperature=0.7
            )
            text = response.choices[0].message.content.strip()
            return Idea(text=text, parent_id=parent_id, generation_depth=depth, elo_rating=BASE_ELO)
        except Exception as e:
            return Idea(text=f"Error generating idea: {str(e)}", parent_id=parent_id, generation_depth=depth, elo_rating=BASE_ELO)

    async def propose_paths(self, n: int = 5):
        """Generates N parallel ideas based on the current top idea or base context."""
        context_str = f"Riddle: {self.state.description}\n"
        if self.state.image_descriptions:
            context_str += "Visual Clues:\n" + "\n".join(self.state.image_descriptions) + "\n"

        parent_id = self.state.top_idea_id
        if parent_id and parent_id in self.state.ideas:
            context_str += f"\nPrevious Reasoning (Immutable Checkpoint):\n{self.state.ideas[parent_id].text}\n"

        depth = self.state.current_generation + 1

        tasks = [self._propose_single_path(context_str, parent_id, depth) for _ in range(n)]
        new_ideas = await asyncio.gather(*tasks)

        for idea in new_ideas:
            self.state.ideas[idea.id] = idea

        self.state.current_generation = depth

    async def _critique_idea(self, idea: Idea, context_str: str):
        """Asks DeepSeek to act as a Critic."""
        try:
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": CRITIC_PROMPT},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nHypothesis:\n{idea.text}\n\nCritique it."}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            import json
            feedback_json = json.loads(response.choices[0].message.content.strip())

            idea.critique = Critique(
                weaknesses=feedback_json.get("weaknesses", []),
                alignment_score=feedback_json.get("alignment_score", 0.5),
                feedback=feedback_json.get("feedback", str(feedback_json))
            )
        except Exception as e:
            idea.critique = Critique(feedback=f"Critique failed: {str(e)}", alignment_score=0.0)

    async def run_critiques(self):
        """Runs critiques on all ideas of the current generation."""
        current_ideas = self.state.get_generation_ideas(self.state.current_generation)
        context_str = f"Riddle: {self.state.description}\n"
        if self.state.image_descriptions:
            context_str += "Visual Clues:\n" + "\n".join(self.state.image_descriptions) + "\n"

        tasks = [self._critique_idea(idea, context_str) for idea in current_ideas]
        await asyncio.gather(*tasks)

    async def _judge_match(self, idea_a: Idea, idea_b: Idea, context_str: str) -> str:
        """Asks DeepSeek Judge to evaluate A vs B."""
        critique_a = idea_a.critique.feedback if idea_a.critique else "No critique available."
        critique_b = idea_b.critique.feedback if idea_b.critique else "No critique available."

        prompt = JUDGE_PROMPT.format(
            context=context_str,
            idea_a=idea_a.text,
            critique_a=critique_a,
            idea_b=idea_b.text,
            critique_b=critique_b
        )
        try:
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip().upper()
            return 'A' if 'A' in result else 'B' if 'B' in result else 'A' # Default to A if unclear
        except Exception:
            return 'A'

    def update_elo(self, winner: Idea, loser: Idea):
        """Standard Elo calculation."""
        expected_winner = winner.expected_score(loser.elo_rating)
        expected_loser = loser.expected_score(winner.elo_rating)

        winner.elo_rating += K_FACTOR * (1.0 - expected_winner)
        loser.elo_rating += K_FACTOR * (0.0 - expected_loser)

    async def run_elo_tournament(self):
        """Runs a pairwise comparison tournament among the current generation."""
        current_ideas = self.state.get_generation_ideas(self.state.current_generation)
        if len(current_ideas) < 2:
            return

        context_str = f"Riddle: {self.state.description}\n"
        if self.state.image_descriptions:
            context_str += "Visual Clues:\n" + "\n".join(self.state.image_descriptions) + "\n"

        # Create pairs (Round Robin style for thoroughness, can be optimized to Swiss for scale)
        pairs = list(itertools.combinations(current_ideas, 2))

        # Process in batches to respect rate limits
        for i in range(0, len(pairs), 5):
            batch = pairs[i:i+5]
            tasks = [self._judge_match(a, b, context_str) for a, b in batch]
            results = await asyncio.gather(*tasks)

            for (a, b), result in zip(batch, results):
                if result == 'A':
                    self.update_elo(a, b)
                else:
                    self.update_elo(b, a)

    def branch_next_generation(self):
        """Selects the top idea to become the checkpoint for the next generation."""
        current_ideas = self.state.get_generation_ideas(self.state.current_generation)
        if not current_ideas:
            return

        # Sort descending by Elo
        current_ideas.sort(key=lambda x: x.elo_rating, reverse=True)
        top_idea = current_ideas[0]
        self.state.top_idea_id = top_idea.id
        return top_idea
