import asyncio
import itertools
import numpy as np
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

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2:
            return 0.0
        vec1 = np.array(v1)
        vec2 = np.array(v2)
        norm_v1 = np.linalg.norm(vec1)
        norm_v2 = np.linalg.norm(vec2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm_v1 * norm_v2))

    async def _embed_text(self, text: str) -> List[float]:
        try:
            response = await asyncio.to_thread(
                gemini_client.models.embed_content,
                model='text-embedding-004',
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            return []

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

    async def _improve_idea(self, idea: Idea, context_str: str) -> Idea:
        """Asks DeepSeek to improve an idea based on its critique."""
        try:
            critique_text = idea.critique.feedback if idea.critique else "No feedback available."
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are the Refiner Agent. Improve the given hypothesis based on the provided critique and context. Return ONLY the improved hypothesis text."},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nOriginal Hypothesis:\n{idea.text}\n\nCritique:\n{critique_text}\n\nProvide the improved hypothesis:"}
                ],
                temperature=0.7
            )
            improved_text = response.choices[0].message.content.strip()
            return Idea(text=improved_text, parent_id=idea.parent_id, generation_depth=idea.generation_depth, elo_rating=idea.elo_rating)
        except Exception as e:
            return idea

    async def _merge_ideas(self, idea1: Idea, idea2: Idea, context_str: str) -> Idea:
        """Asks DeepSeek to synthesize two similar ideas into one unified idea."""
        try:
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are the Synthesizer Agent. Merge the two given similar hypotheses into a single, unified, and comprehensive hypothesis based on the provided context. Return ONLY the unified hypothesis text."},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nHypothesis 1:\n{idea1.text}\n\nHypothesis 2:\n{idea2.text}\n\nProvide the unified hypothesis:"}
                ],
                temperature=0.5
            )
            merged_text = response.choices[0].message.content.strip()
            return Idea(text=merged_text, parent_id=idea1.parent_id, generation_depth=idea1.generation_depth, elo_rating=max(idea1.elo_rating, idea2.elo_rating))
        except Exception as e:
            return idea1 # Fallback to idea1 if merge fails

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

    async def _generate_and_refine_idea(self, context_str: str, parent_id: str, depth: int, log_callback) -> Idea:
        """Proposes an idea and loops to critique and improve it up to 3 times."""
        if log_callback:
            log_callback(f"Proposing new idea for generation {depth}...")
        idea = await self._propose_single_path(context_str, parent_id, depth)

        for i in range(3):
            if log_callback:
                log_callback(f"Critiquing idea (Attempt {i+1}/3)...")
            await self._critique_idea(idea, context_str)

            alignment = idea.critique.alignment_score if idea.critique else 0.0
            if alignment >= 0.8:
                if log_callback:
                    log_callback(f"Idea meets alignment threshold ({alignment:.2f} >= 0.8).")
                break

            if log_callback:
                log_callback(f"Alignment {alignment:.2f} < 0.8. Improving idea...")
            idea = await self._improve_idea(idea, context_str)

        return idea

    async def run_auto_cycle(self, n: int = 5, log_callback=None):
        """Orchestrates the automatic generation, refinement, vector space merging, and evaluation cycle."""
        context_str = f"Riddle: {self.state.description}\n"
        if self.state.image_descriptions:
            context_str += "Visual Clues:\n" + "\n".join(self.state.image_descriptions) + "\n"

        parent_id = self.state.top_idea_id
        if parent_id and parent_id in self.state.ideas:
            context_str += f"\nPrevious Reasoning (Immutable Checkpoint):\n{self.state.ideas[parent_id].text}\n"

        depth = self.state.current_generation + 1

        if log_callback:
            log_callback(f"Starting auto cycle for generation {depth} with {n} parallel paths...")

        # 1. Generate and refine ideas
        tasks = [self._generate_and_refine_idea(context_str, parent_id, depth, log_callback) for _ in range(n)]
        raw_ideas = await asyncio.gather(*tasks)

        # 2. Embed ideas for vector space merging
        if log_callback:
            log_callback("Embedding ideas for vector space comparison...")
        embeddings = await asyncio.gather(*(self._embed_text(idea.text) for idea in raw_ideas))

        # 3. Merge similar ideas
        if log_callback:
            log_callback("Comparing and merging similar ideas...")
        merged_ideas = []
        skip_indices = set()

        for i in range(len(raw_ideas)):
            if i in skip_indices:
                continue

            current_idea = raw_ideas[i]
            current_emb = embeddings[i]

            for j in range(i + 1, len(raw_ideas)):
                if j in skip_indices:
                    continue

                similarity = self._cosine_similarity(current_emb, embeddings[j])
                if similarity > 0.85:
                    if log_callback:
                        log_callback(f"Merging similar ideas ({similarity:.2f} similarity)...")
                    current_idea = await self._merge_ideas(current_idea, raw_ideas[j], context_str)
                    skip_indices.add(j)
                    # Update embedding for the newly merged idea
                    current_emb = await self._embed_text(current_idea.text)

            merged_ideas.append(current_idea)

        # Ensure all final ideas have a critique before evaluation
        if log_callback:
            log_callback("Running final critics on merged/unique ideas...")
        await asyncio.gather(*(self._critique_idea(idea, context_str) for idea in merged_ideas))

        # Save ideas to state
        for idea in merged_ideas:
            self.state.ideas[idea.id] = idea

        self.state.current_generation = depth

        # 4. Run Elo tournament
        if log_callback:
            log_callback(f"Running Elo tournament on {len(merged_ideas)} final ideas...")
        await self.run_elo_tournament(log_callback)

        if log_callback:
            log_callback("Auto cycle complete. Ready for next loop.")

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

    async def run_elo_tournament(self, log_callback=None):
        """Runs a pairwise comparison tournament among the current generation."""
        current_ideas = self.state.get_generation_ideas(self.state.current_generation)
        if len(current_ideas) < 2:
            if log_callback:
                log_callback("Not enough ideas to run Elo tournament.")
            return

        context_str = f"Riddle: {self.state.description}\n"
        if self.state.image_descriptions:
            context_str += "Visual Clues:\n" + "\n".join(self.state.image_descriptions) + "\n"

        # Create pairs (Round Robin style for thoroughness, can be optimized to Swiss for scale)
        pairs = list(itertools.combinations(current_ideas, 2))

        # Process in batches to respect rate limits
        for i in range(0, len(pairs), 5):
            batch = pairs[i:i+5]
            if log_callback:
                log_callback(f"Judging matches {i+1} to {min(i+5, len(pairs))} of {len(pairs)}...")
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
