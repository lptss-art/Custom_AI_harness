# 🧩 Système Multi-Agents de Résolution d'Énigmes (S.M.A.R.E.) - RiddleNexus

L'objectif est de développer une architecture IA autonome capable de décomposer, d'analyser, de chercher des pistes de solution et de valider les étapes d'une chasse au trésor complexe non linéaire, tout en garantissant une traçabilité totale pour permettre le retour en arrière (backtracking) en cas de fausse piste.

## 🧠 Architecture des Agents & Rôles

Le système est articulé autour de plusieurs rôles d'agents principaux, orchestrés par un état global partagé.

### 1. L'Agent Superviseur (Orchestrateur)
* **Mission :** Cartographier la chasse sous forme de Graphe Acyclique Dirigé (DAG). Il exécute la boucle principale `run_auto_cycle`.
* **Entrées :** Énigmes brutes (textes, images, indices).
* **Sorties :** Découpage en briques élémentaires (Pit Stops) et gestion des dépendances.
* **Outil clé :** Registre des "Faits Établis" (Mémoire partagée).

### 2. L'Agent Générateur d'Idées (Créatif)
* **Mission :** Briser la pensée linéaire et générer une grande diversité d'hypothèses.
* **Système Prompt :**
  > "You are the Proposer Agent in a multi-agent system solving a treasure hunt riddle.
  > Your goal is to generate a unique, highly logical, and concrete hypothesis or reasoning path.
  > Do not hallucinate. Base your reasoning STRICTLY on the given context, visual descriptions, and the parent reasoning path (if any).
  > If you need specific, missing details from the visual clues, you may output exactly 'VISUAL_QUERY: <your question about the image>' as your response. The system will look at the image and provide the answer.
  > Otherwise, provide only the reasoning text. Be direct and concise."

### 3. L'Avocat du Diable (Vérificateur & Critique)
* **Mission :** Éliminer impitoyablement les hallucinations et les failles logiques. Il scrute chaque hypothèse. Si le score total d'une hypothèse est jugé trop bas (score < 16.0 / 40.0), il marque la piste comme "Fausse Piste", ce qui déclenche le mécanisme de Backtracking.
* **Système Prompt :**
  > "You are the Critic Agent. Your job is to rigorously cross-examine the given hypothesis.
  > Look for logical flaws, historical inaccuracies, cryptographical errors, and inconsistencies with the provided visual hints.
  > If you need specific, missing details from the visual clues to evaluate this properly, you may output exactly 'VISUAL_QUERY: <your question about the image>' anywhere in your response.
  > Otherwise, you must return a JSON-like structure outlining the weaknesses and a detailed feedback string."

### 4. L'Agent Cartographe (Mémoire Vectorielle)
* **Mission :** Éviter la redondance et le surcalcul en s'assurant qu'une fausse piste n'est pas explorée deux fois.
* **Fonctionnement :** Compare chaque nouvelle idée avec la base de données (ChromaDB) des fausses pistes déjà rejetées via des similarités de plongement (embeddings). Si la distance cosinus est trop faible (ex: < 0.2), l'idée est immédiatement jetée.
* **Système Prompt de Déduplication Symbolique (L'Abstracteur) :**
  > "You are the Abstractor Agent. Extract strict facts from the hypothesis. Return ONLY a JSON object with keys: 'location', 'method', 'key'."

### 5. L'Arbitre des Pistes (Évaluateur Multi-Critères)
* **Mission :** Noter de manière déterministe l'hypothèse pour décider de son sort.
* **Système Prompt :**
  > "You are a panel of expert judges (Cryptography, History, Geography, Logic).
  > Evaluate the following hypothesis based on the context.
  > Return a JSON object with scores from 0 to 10 for each of these keys:
  > - cryptography: The correctness of any cipher or decoding logic.
  > - history: The accuracy of historical references.
  > - geography: The spatial logic and map alignment.
  > - logic: The overall consistency and deductive reasoning."


## ⚙️ Spécifications Techniques & Traçabilité

Chaque piste est enregistrée et tracée selon un modèle de données Pydantic strict (`PisteResolution`) qui embarque à la fois l'idée d'origine (`hypothese_de_depart`), le verdict de l'Avocat du Diable (`analyse_avocat_du_diable` via l'objet `Critique`), et l'évaluation de l'Arbitre (`score_elo` via `ScoreGrid`).

**Mécanisme de Backtracking (Gestion des Fausses Pistes) :**
Lorsqu'un nœud mène à une conclusion impossible (ex: score très faible par l'Arbitre) :
1. **Drapeau d'Échec :** L'Avocat du Diable passe le statut à "Fausse Piste".
2. **Propagation Récursive :** La méthode `backtrack_piste` parcourt la liste des enfants. Toutes les sous-pistes descendantes voient leur statut modifié en "Bloquée par Parent".
3. **Mise à jour de la Mémoire :** L'hypothèse échouée est envoyée à ChromaDB (Cartographe) pour bloquer définitivement cette branche sémantique.
4. **Sélection du Prochain Nœud :** La méthode `branch_next_generation` ignore toute feuille ayant les statuts "Fausse Piste" ou "Bloquée par Parent", et reprend l'exploration à partir du nœud encore ouvert le mieux noté.


## 🛠️ Installation

### Prérequis
*   Python 3.11+
*   A DeepSeek API Key
*   A Google Gemini API Key

### Configuration

1. **Clone the repository and navigate to the directory.**
2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment Variables:**
   Create a `.env` file in the root of the project and add your API keys:
   ```env
   DEEPSEEK_API_KEY="your_deepseek_api_key_here"
   GEMINI_API_KEY="your_gemini_api_key_here"
   ```

## 🚀 Usage

RiddleNexus is operated via a clean, interactive Streamlit interface.

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```
