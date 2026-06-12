# 🧩 Système Multi-Agents de Résolution d'Énigmes (S.M.A.R.E.) - RiddleNexus

L'objectif est de développer une architecture IA autonome capable de décomposer, d'analyser, de chercher des pistes de solution et de valider les étapes d'une chasse au trésor complexe non linéaire, tout en garantissant une traçabilité totale pour permettre le retour en arrière (backtracking) en cas de fausse piste.

## 🧠 Architecture des Agents & Rôles

Le système est articulé autour de 5 rôles d'agents principaux, orchestrés par un état global partagé.

### 1. L'Agent Superviseur (Orchestrateur)
* **Mission :** Cartographier la chasse sous forme de Graphe Acyclique Dirigé (DAG).
* **Entrées :** Énigmes brutes (textes, images, indices).
* **Sorties :** Découpage en briques élémentaires (Pit Stops) et gestion des dépendances.
* **Outil clé :** Registre des "Faits Établis" (Mémoire partagée).

### 2. L'Agent Générateur d'Idées (Créatif)
* **Mission :** Briser la pensée linéaire et générer une grande diversité d'hypothèses via du multi-prompting.
* **Configuration :** Exécution en parallèle de 3 sous-personas :
    * L'Expert Crypto/Logique (chiffrements, anagrammes, mathématiques).
    * L'Historien/Géographe (toponymie, cartographie, contexte culturel).
    * Le Penseur Latéral / Absurde (métaphores, doubles sens, associations incongrues).

### 3. L'Agent Cartographe (Mémoire & Économie)
* **Mission :** Éviter la redondance et le surcalcul.
* **Fonctionnement :** Compare chaque nouvelle idée générée avec la base de données des pistes déjà testées, validées ou rejetées.
* **Outil clé :** Base de données vectorielle (ChromaDB - calcul de similarité sémantique). Si le score de proximité avec un échec passé est trop élevé, l'idée est rejetée.

### 4. Les Agents "Solvers" (Spécialistes Dynamiques)
* **Mission :** Creuser une piste spécifique de manière isolée.
* **Fonctionnement :** Instanciés dynamiquement (spawning) pour chaque piste validée par le Cartographe.
* **Outils dédiés :** Sandbox d'exécution de code Python.

### 5. L'Avocat du Diable & Vérificateur Formel (Sécurité)
* **Mission :** Éliminer impitoyablement les hallucinations.
* **Vérification Formelle :** Exige que le Solver fournisse un script Python fonctionnel (approche Program-of-Thought). Déclenche le Backtracking en cas d'échec formel.

### 6. L'Arbitre des Pistes (Classement Évolutif)
* **Mission :** Organiser des tournois entre les pistes et gérer les priorités d'exécution via un score Elo.


## ⚙️ Spécifications Techniques & Traçabilité

Chaque piste est impérativement enregistrée et tracée selon un modèle de données Pydantic strict (`PisteResolution`) pour permettre le lignage.

**Mécanisme de Backtracking (Gestion des Fausses Pistes) :**
Lorsqu'un nœud du graphe ou une piste maîtresse mène à une conclusion impossible, le système exécute une procédure de nettoyage en cascade :
1. **Drapeau d'Échec :** L'Avocat du Diable passe le statut de la piste incriminée à "Fausse Piste".
2. **Propagation Récursive :** Un script parcourt la liste des pistes_enfants. Toutes les sous-pistes descendantes voient leur statut modifié en "Bloquée par Parent".
3. **Mise à jour de la Mémoire :** L'hypothèse échouée est envoyée à la base de données du Cartographe comme "Erreur Absolue" pour bloquer définitivement cette branche sémantique.
4. **Restauration :** Le Superviseur recharge le dernier état valide du graphe.


## 🛠️ Installation

### Prerequisites
*   Python 3.11+
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
   GEMINI_API_KEY="your_gemini_api_key_here"
   ```

## 🚀 Usage

RiddleNexus is operated via a clean, interactive Streamlit interface.

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```
