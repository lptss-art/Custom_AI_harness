# 🧩 RiddleNexus

RiddleNexus is a multi-agent, multi-modal Python framework designed specifically to solve complex treasure hunt riddles (involving cryptography, historical deductions, and visual mapping).

Inspired by the architectures of **DeepMind's Co-Scientist (Nature 2026)** and the reasoning trees of **AlphaProof Nexus**, RiddleNexus prevents AI hallucinations by forcing agents to cross-examine ideas in a simulated laboratory environment, ranking them via a **Multi-Criteria Scoring Grid** and exploring them using **Backtracking Search**.

## 🧠 How It Works

RiddleNexus operates using a Dual-API routing system and a highly automated reasoning loop driven by distinct agent personas:

*   **Active Vision Loop:** Uses the Google Gemini API to ingest images. Proposer and Critic agents can halt reasoning to output dynamic VISUAL_QUERY requests to interrogate map details on-the-fly instead of relying solely on a static summary.
*   **Automated Propose-Critique-Improve Cycle:**
    *   **The Proposer Agent (DeepSeek)** generates $N$ parallel, independent reasoning paths based on the known clues.
    *   **The Critic Agent (DeepSeek)** rigorously reviews each proposed path, outputting structured feedback regarding logic, flaws, or inconsistencies.
    *   **The Refiner Agent (DeepSeek)** iteratively takes the original idea and the Critic's feedback, actively rewriting the hypothesis to improve it until its multi-criteria score meets a high threshold (looping up to 3 times per idea).
*   **Symbolic Abstraction (Abstractor Agent):** To prevent reasoning fragmentation, RiddleNexus maps all refined ideas into structured dictionaries of strict facts (location, method, key). Hypotheses with exactly matching facts are mathematically proven identical, avoiding the pitfalls of semantic embeddings where opposing logic looks similar. Identical ideas are merged by the Synthesizer.
*   **Multi-Criteria Scoring:** Replaces arbitrary Elo duels. A specialized evaluation maps each hypothesis to a ScoreGrid (Cryptography, History, Geography, Logic) calculating an exact total_score in O(N) time.
*   **Backtracking Search (DFS/MCTS-like):** The hypothesis with the highest total score among all unexpanded leaf nodes across the entire thought tree is selected, allowing the system to backtrack if current scores collapse.

## 🤖 Agent Personas & System Prompts

RiddleNexus leverages specialized DeepSeek agents, each guided by precise system prompts to enforce strict roles within the automated reasoning cycle:

### 1. The Proposer Agent
**Role:** Generates logical hypotheses based on clues and context.
**Prompt:**
> "You are the Proposer Agent in a multi-agent system solving a treasure hunt riddle.
> Your goal is to generate a unique, highly logical, and concrete hypothesis or reasoning path.
> Do not hallucinate. Base your reasoning STRICTLY on the given context, visual descriptions, and the parent reasoning path (if any).
> If you need specific, missing details from the visual clues, you may output exactly 'VISUAL_QUERY: <your question about the image>' as your response. The system will look at the image and provide the answer.
> Otherwise, provide only the reasoning text. Be direct and concise."

### 2. The Critic Agent
**Role:** Rigorously cross-examines proposed hypotheses.
**Prompt:**
> "You are the Critic Agent. Your job is to rigorously cross-examine the given hypothesis.
> Look for logical flaws, historical inaccuracies, cryptographical errors, and inconsistencies with the provided visual hints.
> If you need specific, missing details from the visual clues to evaluate this properly, you may output exactly 'VISUAL_QUERY: <your question about the image>' anywhere in your response.
> Otherwise, you must return a JSON-like structure outlining the weaknesses and a detailed feedback string."

### 3. The Abstractor Agent
**Role:** Translates raw reasoning text into structured, strict facts to enable exact-match deduplication.
**Prompt:**
> "You are the Abstractor Agent. Extract strict facts from the hypothesis. Return ONLY a JSON object with keys: 'location', 'method', 'key'."

### 4. The Multi-Criteria Judge
**Role:** Replaces arbitrary Elo duels by assigning objective scores (0-10) across four specialized domains: Cryptography, History, Geography, and Logic.
**Prompt Concept:** Evaluates the hypothesis and context, returning a structured `ScoreGrid` JSON object determining the idea's exact `total_score` (out of 40).


## ⚙️ Detailed Engine Logic

The `run_auto_cycle` execution pipeline operates as follows:

1. **Generation & Active Vision:** The engine spins up N parallel Proposer agents. During generation, if an agent outputs a `VISUAL_QUERY`, the engine pauses the agent, queries the Gemini Vision model with the specific question against the raw uploaded images, injects the answer into the context, and re-prompts the agent (up to 3 loops).
2. **Critique & Refinement:** Each idea is reviewed by the Critic. If the multi-criteria `total_score` is below the 80% threshold (32/40), the Refiner attempts to rewrite and improve the idea based on the Critic's feedback (up to 3 loops).
3. **Symbolic Deduplication:** Final ideas are passed to the Abstractor. Ideas that map to the exact same dictionary of facts (location, method, key) are considered mathematically identical and are merged.
4. **Scoring:** The unique, merged ideas undergo a final Multi-Criteria Evaluation.
5. **Backtracking Search:** When advancing to the next generation, the engine evaluates *all* unexpanded leaf nodes across the entire reasoning tree (DFS/MCTS-style). It selects the node with the highest `total_score` globally, allowing the system to naturally abandon failing branches and backtrack to a promising past idea.

## 🛠️ Installation

### Prerequisites
*   Python 3.9+
*   A DeepSeek API Key
*   A Google Gemini API Key

### Setup

1. **Clone the repository and navigate to the directory.**

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root of the project and add your API keys:
   ```env
   DEEPSEEK_API_KEY="your_deepseek_api_key_here"
   # DEEPSEEK_BASE_URL="https://api.deepseek.com/v1" # Optional, defaults to this
   GEMINI_API_KEY="your_gemini_api_key_here"
   ```

## 🚀 Usage

RiddleNexus is operated via a clean, interactive Streamlit interface.

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Initialize Context:**
   * Enter the main riddle text in the sidebar.
   * Upload auxiliary text files (`.txt`, `.md`) or image clues (`.png`, `.jpg`).
   * Click "Analyze Images with Gemini" to extract visual data into the context.

3. **Run the Nexus Loop:**
   Interact with the reasoning tree in the UI:
   * **Run Automatic Generation Cycle:** Click this button to launch the automated engine. The UI will stream real-time logs indicating progress as the engine generates, critiques, refines, vectorizes, merges, and scores the ideas.
   * **Review Leaderboard:** Once the automated cycle completes, expand the current ideas to view their reasoning, critic feedback, and final Total Score.
   * **Branch & Iterate Next Loop:** Click this button to manually lock in the top-ranked idea as fact. This advances the depth of reasoning, waiting for you to run the next cycle.
