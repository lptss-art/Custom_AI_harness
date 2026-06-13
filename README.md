# 🧩 Système Multi-Agents de Résolution d'Énigmes (S.M.A.R.E.) - RiddleNexus

L'objectif est de développer une architecture IA autonome capable de décomposer, d'analyser, de chercher des pistes de solution et de valider les étapes d'une chasse au trésor complexe non linéaire, tout en garantissant une traçabilité totale pour permettre le retour en arrière (backtracking) en cas de fausse piste.

## 🧠 Architecture des Agents & Rôles

Le système est articulé autour de plusieurs rôles d'agents principaux, orchestrés par un état global partagé.

### 1. L'Agent Superviseur (Orchestrateur)
* **Mission :** Cartographier la chasse sous forme de Graphe Acyclique Dirigé (DAG). Il exécute la boucle principale `run_auto_cycle`.
* **Entrées :** Énigmes brutes (textes, images, indices).
* **Sorties :** Découpage en briques élémentaires (Pit Stops) et gestion des dépendances.
* **Outil clé :** Registre des "Faits Établis" (Mémoire partagée).

### 2. L'Agent Générateur d'Idées (Créatif & Multi-Prompt)
* **Mission :** Briser la pensée linéaire et générer une grande diversité d'hypothèses en s'appuyant sur plusieurs approches (Logique, Thématique, Latérale).
* **Contrainte de format :** Afin de garantir des itérations claires et cumulatives, chaque hypothèse générée **doit tenir en une seule phrase courte et concise**.
* **Systèmes Prompts :**
  L'agent tourne selon une approche *Round-Robin* sur les prompts suivants :
  - **Logical Proposer :** "Generate a unique, highly logical, and concrete hypothesis based STRICTLY on deductive reasoning..."
  - **Thematic Proposer :** "Generate a hypothesis that deeply aligns with the narrative, history, and theme of the riddle..."
  - **Lateral-Thinking Proposer :** "Generate an improbable, out-of-the-box, or lateral thinking hypothesis..."

### 3. Le Solver de Piste (L'Exécuteur & Sandbox)
* **Mission :** Tester de façon concrète les hypothèses générées à l'aide d'un processus de raisonnement (Chain of Thought), exécuter du code, et fournir un résultat exploitable (`SIMPLE_OUTPUT`).
* **Place dans le cycle :** Générateur d'idées ──► Cartographe (ChromaDB) ──► SOLVER DE PISTE ──► Avocat du Diable ──► Arbitre.
* **Process Interne (CoT, Scripting & Sandbox) :**
  Pour ne pas dériver, cet agent utilise une *Chain of Thought* (CoT) interne couplée à un environnement d'exécution contraint.
  1. **Réflexion (CoT) :** "Pour tester cette hypothèse, je dois d'abord extraire le texte crypté de la page 4, puis appliquer l'algorithme."
  2. **Appel d'outils (Scripting & Sandbox) :** L'agent génère un script Python pour valider de façon computationnelle son hypothèse (ex. casser un code, mathématiques).
     * **Sécurité & Sandbox :** L'exécution du code généré par l'IA se fait via `subprocess.run` encapsulé par un script utilitaire (`sandbox_wrapper.py`). Ce wrapper tente d'appliquer des limites strictes (5 secondes de temps CPU, 256 MB de RAM via le module `resource` sous Unix, et la falsification des variables d'environnement réseau) afin de prévenir les boucles infinies ou les fuites de mémoire. *Attention : Sous Windows, ces limites systèmes ne sont pas applicables et l'exécution se fait uniquement avec un Timeout standard (10s).*
  3. **Résultat (`SIMPLE_OUTPUT`) :** L'agent parse la sortie standard du script (STDOUT) pour produire un livrable final stocké dans la propriété `output_simple` de la piste (par exemple, le texte décrypté). Ce résultat est automatiquement réinjecté dans le contexte pour l'itération suivante, permettant l'enchaînement de tâches complexes (ex: double déchiffrage).

### 4. L'Avocat du Diable (Vérificateur & Critique)
* **Mission :** Éliminer impitoyablement les hallucinations et les failles logiques. Il scrute chaque hypothèse. Si le score total d'une hypothèse est jugé trop bas (score < 16.0 / 40.0), il marque la piste comme "Fausse Piste", ce qui déclenche le mécanisme de Backtracking.
* **Système Prompt :**
  > "You are the Critic Agent. Your job is to rigorously cross-examine the given hypothesis.
  > Look for logical flaws, historical inaccuracies, cryptographical errors, and inconsistencies with the provided visual hints.
  > If you need specific, missing details from the visual clues to evaluate this properly, you may output exactly 'VISUAL_QUERY: <your question about the image>' anywhere in your response.
  > Otherwise, you must return a JSON-like structure outlining the weaknesses and a detailed feedback string."

### 5. L'Agent Cartographe (Mémoire Vectorielle)
* **Mission :** Éviter la redondance et le surcalcul en s'assurant qu'une fausse piste n'est pas explorée deux fois.
* **Fonctionnement :** Compare chaque nouvelle idée avec la base de données (ChromaDB) des fausses pistes déjà rejetées via des similarités de plongement (embeddings). Si la distance cosinus est trop faible (ex: < 0.2), l'idée est immédiatement jetée.
* **Système Prompt de Déduplication Symbolique (L'Abstracteur) :**
  > "You are the Abstractor Agent. Extract strict facts from the hypothesis. Return ONLY a JSON object with keys: 'location', 'method', 'key'."

### 6. L'Arbitre des Pistes (Évaluateur Multi-Critères)
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

**Stratégie de Branchement & Backtracking :**
Lorsqu'un nœud mène à une conclusion impossible (ex: score très faible par l'Arbitre) :
1. **Drapeau d'Échec :** L'Avocat du Diable passe le statut à "Fausse Piste".
2. **Propagation Récursive :** La méthode `backtrack_piste` parcourt la liste des enfants. Toutes les sous-pistes descendantes voient leur statut modifié en "Bloquée par Parent".
3. **Mise à jour de la Mémoire :** L'hypothèse échouée est envoyée à ChromaDB (Cartographe) pour bloquer définitivement cette branche sémantique.

Afin de ne pas s'enfermer dans un tunnel de raisonnement, **le système ne génère pas 100% de ses nouvelles idées à partir d'une seule et même piste.** Lors d'un nouveau cycle, le système sélectionne un mix diversifié de pistes parentes :
- La meilleure piste actuelle.
- Des "runners-up" (autres pistes actives avec de bons scores).
- De potentielles nouvelles approches (retour à la racine).


## 🛠️ Installation

### Prérequis
*   Python 3.11+
*   Le paquet système `graphviz` (pour le rendu de l'arbre de raisonnement)
*   A DeepSeek API Key
*   A Google Gemini API Key

### Configuration

1. **Clone the repository and navigate to the directory.**
2. **Install System Dependencies (Ubuntu/Debian):**
   ```bash
   sudo apt-get update && sudo apt-get install -y graphviz
   ```
3. **Install Python dependencies:**
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

RiddleNexus is operated via a clean, interactive Streamlit interface which features:
*   **Visual Reasoning Tree:** A Graphviz rendered directional diagram showing the evolution of hypotheses.
*   **Parameterization:** Allows configuring the number of ideas/pistes generated per reasoning cycle.
*   **Deep JSON Visibility:** Expanding views of the entire Riddle State or individual reasoning tracks for deep observability.
*   **Architecture Flow Tracker:** A visual banner tracking the active step in the S.M.A.R.E. logic.

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```
