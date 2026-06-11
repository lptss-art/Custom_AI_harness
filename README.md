# 🧩 RiddleNexus

RiddleNexus is a multi-agent, multi-modal Python framework designed specifically to solve complex treasure hunt riddles (involving cryptography, historical deductions, and visual mapping).

Inspired by the architectures of **DeepMind's Co-Scientist (Nature 2026)** and the reasoning trees of **AlphaProof Nexus**, RiddleNexus prevents AI hallucinations by forcing agents to cross-examine ideas in a simulated laboratory environment, ranking them via a **Multi-Criteria Scoring Grid** and exploring them using **Backtracking Search**.

## 🧠 How It Works

RiddleNexus operates using a Dual-API routing system and a highly automated reasoning loop driven by distinct agent personas:

*   **Multimodal Input Processing (Gemini):** Uses the Google Gemini API (via `google-genai`) to ingest images, maps, and visual ciphers, extracting their contents into a textual context.
*   **Automated Propose-Critique-Improve Cycle:**
    *   **The Proposer Agent (DeepSeek)** generates $N$ parallel, independent reasoning paths based on the known clues.
    *   **The Critic Agent (DeepSeek)** rigorously reviews each proposed path, outputting an alignment score and structured feedback regarding logic, flaws, or inconsistencies.
    *   **The Refiner Agent (DeepSeek)** iteratively takes the original idea and the Critic's feedback, actively rewriting the hypothesis to improve it until it meets a high alignment score threshold (looping up to 3 times per idea).
*   **Vector Space Deduplication & Merging:** To prevent reasoning fragmentation, RiddleNexus maps all refined ideas into a vector space using Gemini's text embedding API (`text-embedding-004`). Ideas with a high cosine similarity (>0.85) are passed to a **Synthesizer Agent (DeepSeek)**, which merges them into a single, comprehensive, unified hypothesis.
*   **The Judge Agent (DeepSeek):** Implements an **Elo Tournament** on the final, unique set of ideas. Competing hypotheses are pitted against each other in pairwise duels. The Judge reads the ideas *and* their final critiques to mathematically rank them via Elo scores.
*   **Iterative Branching:** The hypothesis with the highest Elo rating becomes the immutable baseline checkpoint for the next generation of reasoning, creating a robust "thought tree".

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
