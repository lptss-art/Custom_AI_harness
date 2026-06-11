import asyncio
import itertools
import numpy as np
from typing import List, Tuple
from config import deepseek_client, gemini_client, K_FACTOR, BASE_ELO
from models import Idea, Critique, RiddleState

PROPOSER_PROMPT = """You are the Proposer Agent in a multi-agent system solving a treasure hunt riddle.
Your goal is to generate a unique, highly logical, and concrete hypothesis or reasoning path.
Do not hallucinate. Base your reasoning STRICTLY on the given context, visual descriptions, and the parent reasoning path (if any).
If you need specific, missing details from the visual clues, you may output exactly "VISUAL_QUERY: <your question about the image>" as your response. The system will look at the image and provide the answer.
Otherwise, provide only the reasoning text. Be direct and concise."""

CRITIC_PROMPT = """You are the Critic Agent. Your job is to rigorously cross-examine the given hypothesis.
Look for logical flaws, historical inaccuracies, cryptographical errors, and inconsistencies with the provided visual hints.
If you need specific, missing details from the visual clues to evaluate this properly, you may output exactly "VISUAL_QUERY: <your question about the image>" anywhere in your response.
Otherwise, you must return a JSON-like structure outlining the weaknesses and a detailed feedback string."""

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
        self.raw_images = []

    async def _abstract_idea(self, text: str) -> dict:
        """Abstracts an idea into a strictly comparable dictionary of facts."""
        try:
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are the Abstractor Agent. Extract strict facts from the hypothesis. Return ONLY a JSON object with keys: 'location', 'method', 'key'."},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            import json
            return json.loads(response.choices[0].message.content.strip())
        except Exception:
            return {"location": "unknown", "method": "unknown", "key": "unknown"}

    async def analyze_image_with_gemini(self, image_data: bytes, mime_type: str, query: str = None) -> str:
        """Uses Gemini to extract visual hints from an image."""
        try:
            prompt_text = query if query else "Extract all hidden symbols, text, and visual clues relevant to a treasure hunt riddle from this image."
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    prompt_text,
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
            current_context = f"Context:\n{context_str}\n\nPropose a hypothesis:"
            for _ in range(3): # Allow up to 3 visual queries per proposal
                response = await deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": PROPOSER_PROMPT},
                        {"role": "user", "content": current_context}
                    ],
                    temperature=0.7
                )
                text = response.choices[0].message.content.strip()

                if "VISUAL_QUERY:" in text:
                    query = text.split("VISUAL_QUERY:")[1].strip()
                    answers = []
                    for img_data, mime in getattr(self, 'raw_images', []):
                        ans = await self.analyze_image_with_gemini(img_data, mime, query)
                        answers.append(ans)
                    current_context += f"\n\n[Visual Query: {query}]\n[Answer: {' | '.join(answers)}]\n"
                else:
                    return Idea(text=text, parent_id=parent_id, generation_depth=depth, total_score=0.0)

            return Idea(text=text, parent_id=parent_id, generation_depth=depth, total_score=0.0)
        except Exception as e:
            return Idea(text=f"Error generating idea: {str(e)}", parent_id=parent_id, generation_depth=depth, total_score=0.0)

    async def _generate_and_refine_idea(self, context_str: str, parent_id: str, depth: int, log_callback) -> Idea:
        """Proposes an idea and loops to critique and improve it up to 3 times."""
        if log_callback:
            log_callback(f"Proposing new idea for generation {depth}...")
        idea = await self._propose_single_path(context_str, parent_id, depth)

        for i in range(3):
            if log_callback:
                log_callback(f"Critiquing idea (Attempt {i+1}/3)...")
            await self._critique_idea(idea, context_str)

            # Re-evaluate multi criteria here to check if it's good enough
            await self._evaluate_multi_criteria(idea, context_str)
            total = idea.total_score if idea.total_score else 0.0

            # If out of 40, 32 is 80%
            if total >= 32.0:
                if log_callback:
                    log_callback(f"Idea meets score threshold ({total} >= 32).")
                break

            if log_callback:
                log_callback(f"Score {total} < 32. Improving idea...")
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

        # 2. Abstract ideas for symbolic deduplication
        if log_callback:
            log_callback("Abstracting ideas for exact symbolic comparison...")
        abstractions = await asyncio.gather(*(self._abstract_idea(idea.text) for idea in raw_ideas))
        for idea, abs_facts in zip(raw_ideas, abstractions):
            idea.abstracted_facts = abs_facts

        # 3. Merge similar ideas based on exact abstraction equality
        if log_callback:
            log_callback("Comparing and merging similar ideas...")
        merged_ideas = []
        skip_indices = set()

        for i in range(len(raw_ideas)):
            if i in skip_indices:
                continue

            current_idea = raw_ideas[i]
            current_abs = abstractions[i]

            for j in range(i + 1, len(raw_ideas)):
                if j in skip_indices:
                    continue

                # Check exact equality of abstracted dictionary
                if current_abs == abstractions[j]:
                    if log_callback:
                        log_callback(f"Merging similar ideas (exact symbolic match)...")
                    current_idea = await self._merge_ideas(current_idea, raw_ideas[j], context_str)
                    skip_indices.add(j)
                    # Re-abstract newly merged idea
                    current_idea.abstracted_facts = await self._abstract_idea(current_idea.text)
                    current_abs = current_idea.abstracted_facts

            merged_ideas.append(current_idea)

        # Ensure all final ideas have a critique before evaluation
        if log_callback:
            log_callback("Running final critics on merged/unique ideas...")
        await asyncio.gather(*(self._critique_idea(idea, context_str) for idea in merged_ideas))

        # Save ideas to state
        for idea in merged_ideas:
            self.state.ideas[idea.id] = idea

        self.state.current_generation = depth

        # 4. Evaluate Multi-Criteria Scores
        if log_callback:
            log_callback(f"Evaluating multi-criteria scores for {len(merged_ideas)} final ideas...")
        await asyncio.gather(*(self._evaluate_multi_criteria(idea, context_str) for idea in merged_ideas))

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

    async def _evaluate_multi_criteria(self, idea: Idea, context_str: str):
        """Asks DeepSeek to evaluate the idea based on multiple specialized criteria."""
        prompt = f"""You are a panel of expert judges (Cryptography, History, Geography, Logic).
Evaluate the following hypothesis based on the context.
Context:
{context_str}

Hypothesis:
{idea.text}

Return a JSON object with scores from 0 to 10 for each of these keys:
- cryptography: The correctness of any cipher or decoding logic.
- history: The accuracy of historical references.
- geography: The spatial logic and map alignment.
- logic: The overall consistency and deductive reasoning.
"""
        try:
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": "You output JSON strictly."}, {"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            import json
            scores = json.loads(response.choices[0].message.content.strip())
            from models import ScoreGrid, Critique

            if not idea.critique:
                idea.critique = Critique(feedback="Evaluated multi-criteria.", score_grid=ScoreGrid())

            idea.critique.score_grid = ScoreGrid(
                cryptography=int(scores.get("cryptography", 0)),
                history=int(scores.get("history", 0)),
                geography=int(scores.get("geography", 0)),
                logic=int(scores.get("logic", 0))
            )
            idea.total_score = float(
                idea.critique.score_grid.cryptography +
                idea.critique.score_grid.history +
                idea.critique.score_grid.geography +
                idea.critique.score_grid.logic
            )
        except Exception as e:
            from models import ScoreGrid, Critique
            if not idea.critique:
                idea.critique = Critique(feedback=f"Evaluation failed: {str(e)}", score_grid=ScoreGrid())
            else:
                idea.critique.score_grid = ScoreGrid()
            idea.total_score = 0.0

    async def _critique_idea(self, idea: Idea, context_str: str):
        """Asks DeepSeek to act as a Critic."""
        try:
            current_context = f"Context:\n{context_str}\n\nHypothesis:\n{idea.text}\n\nCritique it."
            for _ in range(3):
                # When looping for visual queries, we temporarily disable JSON format to allow free-text query.
                response = await deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": CRITIC_PROMPT},
                        {"role": "user", "content": current_context}
                    ],
                    temperature=0.3
                )
                text = response.choices[0].message.content.strip()

                if "VISUAL_QUERY:" in text:
                    query = text.split("VISUAL_QUERY:")[1].strip()
                    answers = []
                    for img_data, mime in getattr(self, 'raw_images', []):
                        ans = await self.analyze_image_with_gemini(img_data, mime, query)
                        answers.append(ans)
                    current_context += f"\n\n[Visual Query: {query}]\n[Answer: {' | '.join(answers)}]\n\nPlease provide the final critique in JSON format."
                else:
                    break

            final_response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": CRITIC_PROMPT + " Output STRICTLY in JSON format."},
                    {"role": "user", "content": current_context}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            import json
            feedback_json = json.loads(final_response.choices[0].message.content.strip())

            from models import Critique
            idea.critique = Critique(
                weaknesses=feedback_json.get("weaknesses", []),
                feedback=feedback_json.get("feedback", str(feedback_json))
            )
        except Exception as e:
            from models import Critique
            idea.critique = Critique(feedback=f"Critique failed: {str(e)}")

    async def run_critiques(self):
        """Runs critiques on all ideas of the current generation."""
        current_ideas = self.state.get_generation_ideas(self.state.current_generation)
        context_str = f"Riddle: {self.state.description}\n"
        if self.state.image_descriptions:
            context_str += "Visual Clues:\n" + "\n".join(self.state.image_descriptions) + "\n"

        tasks = [self._critique_idea(idea, context_str) for idea in current_ideas]
        await asyncio.gather(*tasks)


    def branch_next_generation(self):
        """Selects the best open node (leaf) across all generations to become the checkpoint, implementing backtracking."""
        if not self.state.ideas:
            return None

        # Find all parent IDs
        parent_ids = {idea.parent_id for idea in self.state.ideas.values() if idea.parent_id is not None}

        # A leaf node is an idea whose ID is not in parent_ids
        leaf_nodes = [idea for idea in self.state.ideas.values() if idea.id not in parent_ids]

        if not leaf_nodes:
            return None

        # Sort descending by total_score
        leaf_nodes.sort(key=lambda x: x.total_score, reverse=True)
        top_idea = leaf_nodes[0]
        self.state.top_idea_id = top_idea.id
        self.state.current_generation = top_idea.generation_depth
        return top_idea
