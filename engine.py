import chromadb
from chromadb.config import Settings
import asyncio
import itertools
import numpy as np
from typing import List, Tuple
from config import deepseek_client, gemini_client, K_FACTOR, BASE_ELO
from models import PisteResolution, Critique, RiddleState

PROPOSER_PROMPTS = [
    """You are the Logical Proposer Agent in a multi-agent system solving a treasure hunt riddle.
Your goal is to generate a unique, highly logical, and concrete hypothesis based STRICTLY on deductive reasoning from the given clues.
If you need specific, missing details from the visual clues, you may output exactly "VISUAL_QUERY: <your question about the image>".
Otherwise, provide ONLY your hypothesis. IMPORTANT: Your hypothesis MUST be exactly ONE SINGLE, short and concise sentence. Do not add any explanations.""",

    """You are the Thematic Proposer Agent in a multi-agent system solving a treasure hunt riddle.
Your goal is to generate a hypothesis that deeply aligns with the narrative, history, and theme of the riddle. Focus on lore and thematic connections.
If you need specific, missing details from the visual clues, you may output exactly "VISUAL_QUERY: <your question about the image>".
Otherwise, provide ONLY your hypothesis. IMPORTANT: Your hypothesis MUST be exactly ONE SINGLE, short and concise sentence. Do not add any explanations.""",

    """You are the Lateral-Thinking Proposer Agent in a multi-agent system solving a treasure hunt riddle.
Your goal is to generate an improbable, out-of-the-box, or lateral thinking hypothesis that challenges obvious assumptions.
If you need specific, missing details from the visual clues, you may output exactly "VISUAL_QUERY: <your question about the image>".
Otherwise, provide ONLY your hypothesis. IMPORTANT: Your hypothesis MUST be exactly ONE SINGLE, short and concise sentence. Do not add any explanations."""
]

CRITIC_PROMPT = """You are the Critic Agent. Your job is to rigorously cross-examine the given hypothesis.
Look for logical flaws, historical inaccuracies, cryptographical errors, and inconsistencies with the provided visual hints.
If you need specific, missing details from the visual clues to evaluate this properly, you may output exactly "VISUAL_QUERY: <your question about the image>" anywhere in your response.
Otherwise, you must return a JSON-like structure outlining the weaknesses and a detailed feedback string."""

JUDGE_PROMPT = """You are the Elo Tournament Judge.
You are evaluating two competing pistes to solve a riddle.
Context: {context}

Idea A: {piste_a}
Critique of Idea A: {critique_a}

Idea B: {piste_b}
Critique of Idea B: {critique_b}

Compare the two pistes based on logic, consistency with clues, the rigor of their critiques, and likelihood of being the correct step.
Output ONLY 'A' if Idea A is better, or 'B' if Idea B is better."""


class NexusEngine:
    def __init__(self, state: RiddleState):
        self.state = state
        self.raw_images = []
        self.chroma_client = chromadb.Client(Settings(is_persistent=False))
        self.collection = self.chroma_client.get_or_create_collection(name="fausses_pistes")


    def is_fausse_piste_similar(self, text: str) -> bool:
        """Cartographe: Checks if the generated piste is too similar to a known fausse piste."""
        if self.collection.count() == 0:
            return False

        results = self.collection.query(
            query_texts=[text],
            n_results=1
        )
        if results and results['distances'] and results['distances'][0]:
            if results['distances'][0][0] < 0.2:
                return True
        return False

    def add_fausse_piste_to_memory(self, text: str):
        import uuid as _uuid
        self.collection.add(
            documents=[text],
            ids=[str(_uuid.uuid4())]
        )

    def backtrack_piste(self, piste_id: str):
        """Propagation Récursive pour invalider les enfants."""
        piste = self.state.pistes.get(piste_id)
        if not piste:
            return

        for enfant_id in piste.pistes_enfants:
            enfant = self.state.pistes.get(enfant_id)
            if enfant and enfant.statut != "Fausse Piste":
                enfant.statut = "Bloquée par Parent"
                enfant.raison_blocage = f"Parent {piste_id} was marked as Fausse Piste."
                self.backtrack_piste(enfant_id)

    async def run_avocat_du_diable(self, piste: PisteResolution):
        """Vérificateur Formel (Sécurité). Si le score_elo est très bas, flag Fausse Piste."""
        if piste.statut == "Fausse Piste":
            return

        if piste.score_elo < 16.0:  # Less than 40% threshold for critical failure
            piste.statut = "Fausse Piste"
            self.add_fausse_piste_to_memory(piste.hypothese_de_depart)
            self.backtrack_piste(piste.id_piste)

    async def _abstract_piste(self, text: str) -> dict:
        """Abstracts an piste into a strictly comparable dictionary of facts."""
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

    async def _improve_piste(self, piste: PisteResolution, context_str: str) -> PisteResolution:
        """Asks DeepSeek to improve an piste based on its critique."""
        try:
            critique_text = piste.analyse_avocat_du_diable.feedback if piste.analyse_avocat_du_diable else "No feedback available."
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are the Refiner Agent. Improve the given hypothesis based on the provided critique and context. Return ONLY the improved hypothesis text."},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nOriginal Hypothesis:\n{piste.hypothese_de_depart}\n\nCritique:\n{critique_text}\n\nProvide the improved hypothesis:"}
                ],
                temperature=0.7
            )
            improved_text = response.choices[0].message.content.strip()
            piste.hypothese_de_depart = improved_text
            return piste
        except Exception as e:
            return piste

    async def _merge_pistes(self, piste1: PisteResolution, piste2: PisteResolution, context_str: str) -> PisteResolution:
        """Asks DeepSeek to synthesize two similar pistes into one unified piste."""
        try:
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are the Synthesizer Agent. Merge the two given similar hypotheses into a single, unified, and comprehensive hypothesis based on the provided context. Return ONLY the unified hypothesis text."},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nHypothesis 1:\n{piste1.hypothese_de_depart}\n\nHypothesis 2:\n{piste2.hypothese_de_depart}\n\nProvide the unified hypothesis:"}
                ],
                temperature=0.5
            )
            merged_text = response.choices[0].message.content.strip()
            piste1.hypothese_de_depart = merged_text
            piste1.score_elo = max(piste1.score_elo, piste2.score_elo)
            return piste1
        except Exception as e:
            return piste1 # Fallback to piste1 if merge fails

    async def _propose_single_path(self, context_str: str, pistes_parentes_id: str = None, depth: int = 0, prompt_template: str = PROPOSER_PROMPTS[0]) -> PisteResolution:
        """Asks DeepSeek to generate one hypothesis using a specific persona prompt."""
        try:
            current_context = f"Context:\n{context_str}\n\nPropose a hypothesis (remember, ONE single sentence):"
            for _ in range(3): # Allow up to 3 visual queries per proposal
                response = await deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": prompt_template},
                        {"role": "user", "content": current_context}
                    ],
                    temperature=0.8
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
                    return PisteResolution(hypothese_de_depart=text, pistes_parentes=[pistes_parentes_id] if pistes_parentes_id else [], generation_depth=depth, score_elo=0.0)

            return PisteResolution(hypothese_de_depart=text, pistes_parentes=[pistes_parentes_id] if pistes_parentes_id else [], generation_depth=depth, score_elo=0.0)
        except Exception as e:
            return PisteResolution(hypothese_de_depart=f"Error generating piste: {str(e)}", pistes_parentes=[pistes_parentes_id] if pistes_parentes_id else [], generation_depth=depth, score_elo=0.0)

    async def _generate_and_refine_piste(self, context_str: str, pistes_parentes_id: str, depth: int, log_callback, prompt_template: str) -> PisteResolution:
        """Proposes an piste and loops to critique and improve it up to 3 times."""
        if log_callback:
            log_callback(f"Proposing new piste for generation {depth}...")
        piste = await self._propose_single_path(context_str, pistes_parentes_id, depth, prompt_template)
        if piste.pistes_parentes:
            for pid in piste.pistes_parentes:
                if pid in self.state.pistes:
                    self.state.pistes[pid].pistes_enfants.append(piste.id_piste)

        for i in range(3):
            if log_callback:
                log_callback(f"Critiquing piste (Attempt {i+1}/3)...")
            await self._critique_piste(piste, context_str)

            # Re-evaluate multi criteria here to check if it's good enough
            await self._evaluate_multi_criteria(piste, context_str)
            total = piste.score_elo if piste.score_elo else 0.0

            # If out of 40, 32 is 80%
            if total >= 32.0:
                if log_callback:
                    log_callback(f"Idea meets score threshold ({total} >= 32).")
                break

            if log_callback:
                log_callback(f"Score {total} < 32. Improving piste...")
            piste = await self._improve_piste(piste, context_str)

        return piste

    async def run_auto_cycle(self, n: int = 5, log_callback=None):
        """Orchestrates the automatic generation, refinement, vector space merging, and evaluation cycle."""
        base_context_str = f"Riddle: {self.state.description}\n"
        if self.state.image_descriptions:
            base_context_str += "Visual Clues:\n" + "\n".join(self.state.image_descriptions) + "\n"

        depth = self.state.current_generation + 1

        if log_callback:
            log_callback(f"Starting auto cycle for generation {depth} with {n} parallel paths using diverse prompts and mixed parents...")

        # Select a mix of parent pistes to branch from, not just the top one
        parent_candidates = self._get_mixed_parent_candidates()

        # 1. Generate and refine pistes using a round-robin of diverse prompts and selected parents
        tasks = []
        for i in range(n):
            prompt_template = PROPOSER_PROMPTS[i % len(PROPOSER_PROMPTS)]
            parent_piste = parent_candidates[i % len(parent_candidates)] if parent_candidates else None
            parent_id = parent_piste.id_piste if parent_piste else None

            # Build specific context for this generation, including simple output if available
            specific_context = base_context_str
            if parent_piste:
                specific_context += f"\nPrevious Reasoning (Parent Checkpoint):\n{parent_piste.hypothese_de_depart}\n"
                if parent_piste.output_simple:
                    specific_context += f"Result/Modified Riddle from Parent:\n{parent_piste.output_simple}\n"

            tasks.append(self._generate_and_refine_piste(specific_context, parent_id, depth, log_callback, prompt_template))

        raw_pistes = await asyncio.gather(*tasks)

        # 2. Abstract pistes for symbolic deduplication
        if log_callback:
            log_callback("Abstracting pistes for exact symbolic comparison...")
        abstractions = await asyncio.gather(*(self._abstract_piste(piste.hypothese_de_depart) for piste in raw_pistes))
        for piste, abs_facts in zip(raw_pistes, abstractions):
            piste.abstracted_facts = abs_facts

        # 3. Merge similar pistes based on exact abstraction equality
        if log_callback:
            log_callback("Comparing and merging similar pistes...")
        merged_pistes = []
        skip_indices = set()

        for i in range(len(raw_pistes)):
            if i in skip_indices:
                continue

            current_piste = raw_pistes[i]
            current_abs = abstractions[i]

            for j in range(i + 1, len(raw_pistes)):
                if j in skip_indices:
                    continue

                # Check exact equality of abstracted dictionary
                if current_abs == abstractions[j]:
                    if log_callback:
                        log_callback(f"Merging similar pistes (exact symbolic match)...")
                    current_piste = await self._merge_pistes(current_piste, raw_pistes[j], base_context_str)
                    skip_indices.add(j)
                    # Re-abstract newly merged piste
                    current_piste.abstracted_facts = await self._abstract_piste(current_piste.hypothese_de_depart)
                    current_abs = current_piste.abstracted_facts

            merged_pistes.append(current_piste)

        # Ensure all final pistes have a critique before evaluation
        if log_callback:
            log_callback("Running final critics on merged/unique pistes...")
        await asyncio.gather(*(self._critique_piste(piste, base_context_str) for piste in merged_pistes))

        # Save pistes to state
        for piste in merged_pistes:
            self.state.pistes[piste.id_piste] = piste

        self.state.current_generation = depth

        # 4. Evaluate Multi-Criteria Scores
        if log_callback:
            log_callback(f"Evaluating multi-criteria scores for {len(merged_pistes)} final pistes...")
        await asyncio.gather(*(self._evaluate_multi_criteria(piste, base_context_str) for piste in merged_pistes))

        # 5. Avocat du Diable / Cartographe (Early Rejection)
        if log_callback:
            log_callback('Running Cartographe & Avocat du Diable checks to filter bad tracks before execution...')
        for piste in merged_pistes:
            if self.is_fausse_piste_similar(piste.hypothese_de_depart):
                piste.statut = 'Fausse Piste'
                piste.raison_blocage = 'Rejetée par le Cartographe: Similar to a known fausse piste.'
            else:
                await self.run_avocat_du_diable(piste)

        # Filter active pistes
        active_pistes = [p for p in merged_pistes if p.statut not in ['Fausse Piste', 'Bloquée par Parent']]

        # 6. Solver de Piste (L'Exécuteur)
        if active_pistes and log_callback:
            log_callback(f"Running Solver de Piste (L'Exécuteur) on {len(active_pistes)} valid tracks...")
        if active_pistes:
            await asyncio.gather(*(self._run_solver(piste, base_context_str, log_callback) for piste in active_pistes))

        if log_callback:
            log_callback("Auto cycle complete. Ready for next loop.")

    async def _run_solver(self, piste: PisteResolution, context_str: str, log_callback=None):
        """Asks DeepSeek to act as a Solver, generate a Python test script, and extract a simple output."""
        solver_prompt = f"""You are the Solver Agent (L'Exécuteur).
Your goal is to test and verify the given hypothesis using a Chain of Thought.
First, explain your reasoning (CoT) on how to test this hypothesis.
Then, if a computational check is needed (e.g., deciphering text, math, logic validation), provide a single valid Python script enclosed in ```python ... ``` blocks.
Finally, and most importantly, you MUST provide a "SIMPLE_OUTPUT:" line at the very end of your response containing ONLY the deciphered text, key findings, or modified riddle that should be passed on to the next iteration.

Context:
{context_str}

Hypothesis:
{piste.hypothese_de_depart}
"""
        try:
            messages = [
                {"role": "system", "content": "You are a Python executor agent."},
                {"role": "user", "content": solver_prompt}
            ]

            max_retries = 2
            attempts = 0

            import re
            import subprocess
            import tempfile
            import os
            import sys

            while attempts <= max_retries:
                attempts += 1

                response = await deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0.2
                )
                content = response.choices[0].message.content.strip()

                # Extract python code if present
                code_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)

                if not code_match:
                    piste.protocole_de_test = f"Logical Deduction:\n{content}"
                    piste.resultat_du_test = "No script generated."
                    break

                script_code = code_match.group(1)
                piste.protocole_de_test = f"Reasoning (Attempt {attempts}):\n{content}\n\nScript:\n{script_code}"

                try:
                    # Write script to a temporary file
                    fd, temp_path = tempfile.mkstemp(suffix=".py")
                    with os.fdopen(fd, 'w') as f:
                        f.write(script_code)

                    wrapper_path = os.path.join(os.path.dirname(__file__), "sandbox_wrapper.py")

                    # Run the script
                    result = subprocess.run(
                        [sys.executable, wrapper_path, temp_path],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    if result.returncode == 0:
                        piste.resultat_du_test = f"Execution successful.\nStdout:\n{result.stdout.strip()}"
                        break  # Success, exit the retry loop
                    else:
                        error_msg = f"Execution failed (Code {result.returncode}).\nStdout:\n{result.stdout.strip()}\nStderr:\n{result.stderr.strip()}"
                        piste.resultat_du_test = error_msg

                        if attempts <= max_retries:
                            # Feed the error back to the LLM to ask for a fix
                            messages.append({"role": "assistant", "content": content})
                            messages.append({"role": "user", "content": f"Your script failed with the following error:\n{error_msg}\n\nPlease fix the script, rewrite it completely within ```python ``` blocks, and remember to include the SIMPLE_OUTPUT: line."})
                            if log_callback:
                                log_callback(f"Script execution failed for piste {piste.id_piste[:5]}. Retrying (Attempt {attempts}/{max_retries})...")

                except subprocess.TimeoutExpired:
                    piste.resultat_du_test = "Execution timed out (Wall clock limit)."
                    break # Don't retry timeouts
                except Exception as ex:
                    piste.resultat_du_test = f"Execution error: {str(ex)}"
                    break
                finally:
                    # Clean up temp file
                    if 'temp_path' in locals() and os.path.exists(temp_path):
                        os.remove(temp_path)

            # Extract simple output (from the last response content)
            simple_output_match = re.search(r'SIMPLE_OUTPUT:\s*(.*)', content, re.IGNORECASE | re.DOTALL)
            if simple_output_match:
                piste.output_simple = simple_output_match.group(1).strip()
            else:
                piste.output_simple = None

        except Exception as e:
            piste.protocole_de_test = f"Solver failed: {str(e)}"
            piste.resultat_du_test = "Error"
            piste.output_simple = None

    async def propose_paths(self, n: int = 5):
        """Generates N parallel pistes based on the current top piste or base context."""
        context_str = f"Riddle: {self.state.description}\n"
        if self.state.image_descriptions:
            context_str += "Visual Clues:\n" + "\n".join(self.state.image_descriptions) + "\n"

        pistes_parentes_id = self.state.top_piste_id
        if pistes_parentes_id and pistes_parentes_id in self.state.pistes:
            context_str += f"\nPrevious Reasoning (Immutable Checkpoint):\n{self.state.pistes[pistes_parentes_id].hypothese_de_depart}\n"

        depth = self.state.current_generation + 1

        tasks = []
        for i in range(n):
            prompt_template = PROPOSER_PROMPTS[i % len(PROPOSER_PROMPTS)]
            tasks.append(self._propose_single_path(context_str, pistes_parentes_id, depth, prompt_template))

        new_pistes = await asyncio.gather(*tasks)

        for piste in new_pistes:
            self.state.pistes[piste.id_piste] = piste

        self.state.current_generation = depth

    async def _evaluate_multi_criteria(self, piste: PisteResolution, context_str: str):
        """Asks DeepSeek to evaluate the piste based on multiple specialized criteria."""
        prompt = f"""You are a panel of expert judges (Cryptography, History, Geography, Logic).
Evaluate the following hypothesis based on the context.
Context:
{context_str}

Hypothesis:
{piste.hypothese_de_depart}

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

            if not piste.analyse_avocat_du_diable:
                piste.analyse_avocat_du_diable = Critique(feedback="Evaluated multi-criteria.", score_grid=ScoreGrid())

            piste.analyse_avocat_du_diable.score_grid = ScoreGrid(
                cryptography=int(scores.get("cryptography", 0)),
                history=int(scores.get("history", 0)),
                geography=int(scores.get("geography", 0)),
                logic=int(scores.get("logic", 0))
            )
            piste.score_elo = float(
                piste.analyse_avocat_du_diable.score_grid.cryptography +
                piste.analyse_avocat_du_diable.score_grid.history +
                piste.analyse_avocat_du_diable.score_grid.geography +
                piste.analyse_avocat_du_diable.score_grid.logic
            )
        except Exception as e:
            from models import ScoreGrid, Critique
            if not piste.analyse_avocat_du_diable:
                piste.analyse_avocat_du_diable = Critique(feedback=f"Evaluation failed: {str(e)}", score_grid=ScoreGrid())
            else:
                piste.analyse_avocat_du_diable.score_grid = ScoreGrid()
            piste.score_elo = 0.0

    async def _critique_piste(self, piste: PisteResolution, context_str: str):
        """Asks DeepSeek to act as a Critic."""
        try:
            current_context = f"Context:\n{context_str}\n\nHypothesis:\n{piste.hypothese_de_depart}\n\nCritique it."
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
            piste.analyse_avocat_du_diable = Critique(
                weaknesses=feedback_json.get("weaknesses", []),
                feedback=feedback_json.get("feedback", str(feedback_json))
            )
        except Exception as e:
            from models import Critique
            piste.analyse_avocat_du_diable = Critique(feedback=f"Critique failed: {str(e)}")

    async def run_critiques(self):
        """Runs critiques on all pistes of the current generation."""
        current_pistes = self.state.get_generation_pistes(self.state.current_generation)
        context_str = f"Riddle: {self.state.description}\n"
        if self.state.image_descriptions:
            context_str += "Visual Clues:\n" + "\n".join(self.state.image_descriptions) + "\n"

        tasks = [self._critique_piste(piste, context_str) for piste in current_pistes]
        await asyncio.gather(*tasks)


    def _get_mixed_parent_candidates(self) -> List[PisteResolution]:
        """Returns a diverse mix of up to 3 parent pistes: the best one, a random active one, and potentially a blank slate."""
        if not self.state.pistes:
            return []

        pistes_parentes_ids = {pid for piste in self.state.pistes.values() if piste.pistes_parentes for pid in piste.pistes_parentes}
        leaf_nodes = [piste for piste in self.state.pistes.values() if piste.id_piste not in pistes_parentes_ids and piste.statut not in ["Fausse Piste", "Bloquée par Parent"]]

        if not leaf_nodes:
            return []

        leaf_nodes.sort(key=lambda x: x.score_elo, reverse=True)

        candidates = [leaf_nodes[0]] # Always include the best

        if len(leaf_nodes) > 1:
            import random
            # Include a random runner-up
            candidates.append(random.choice(leaf_nodes[1:]))

        # Potentially add a 'None' (fresh start) by adding nothing here, and letting the loop handle it
        return candidates

    def branch_next_generation(self):
        """Selects the best open node (leaf) across all generations to become the checkpoint, implementing backtracking."""
        if not self.state.pistes:
            return None

        # Find all parent IDs
        pistes_parentes_ids = {(piste.pistes_parentes[0] if piste.pistes_parentes else None) for piste in self.state.pistes.values() if (piste.pistes_parentes[0] if piste.pistes_parentes else None) is not None}

        # A leaf node is an piste whose ID is not in pistes_parentes_ids
        leaf_nodes = [piste for piste in self.state.pistes.values() if piste.id_piste not in pistes_parentes_ids and piste.statut not in ["Fausse Piste", "Bloquée par Parent"]]

        if not leaf_nodes:
            return None

        # Sort descending by score_elo
        leaf_nodes.sort(key=lambda x: x.score_elo, reverse=True)
        top_piste = leaf_nodes[0]
        self.state.top_piste_id = top_piste.id_piste
        self.state.current_generation = top_piste.generation_depth
        return top_piste
