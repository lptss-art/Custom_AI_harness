# 🧩 RiddleNexus

RiddleNexus is a multi-agent, multi-modal Python framework designed specifically to solve complex treasure hunt riddles (involving cryptography, historical deductions, and visual mapping).

Inspired by the architectures of **DeepMind's Co-Scientist (Nature 2026)** and the reasoning trees of **AlphaProof Nexus**, RiddleNexus prevents AI hallucinations by forcing agents to cross-examine ideas in a simulated laboratory environment, ranking them mathematically via an **Elo Tournament**.

## 🧠 How It Works

RiddleNexus operates using a Dual-API routing system and three distinct agent personas:

*   **Multimodal Input Processing (Gemini):** Uses the Google Gemini API (via `google-genai`) to ingest images, maps, and visual ciphers, extracting their contents into a textual context.
*   **The Proposer Agent (DeepSeek):** Generates $N$ parallel, independent reasoning paths based on the known clues and the current state of the reasoning tree.
*   **The Critic Agent (DeepSeek):** Rigorously reviews each proposed path, searching for logical flaws, cryptographic errors, or inconsistencies with visual clues. It outputs a structured critique.
*   **The Judge Agent (DeepSeek):** Implements an **Elo Tournament**. Competing hypotheses are pitted against each other in pairwise duels. The Judge reads the ideas *and* their critiques to decide the winner.
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
   Follow the numbered steps in the UI to walk through a generation:
   * **Step 1:** Click *Propose Parallel Hypotheses* to generate ideas.
   * **Step 2:** Click *Run Critics* to generate counter-arguments for each idea.
   * **Step 3:** Click *Run Elo Tournament* to evaluate the ideas and rank them on the leaderboard.
   * **Step 4:** Click *Branch Next Generation* to lock in the top-ranked idea as fact and move to the next depth of reasoning.
