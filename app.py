import streamlit as st
import asyncio
import graphviz
from models import RiddleState
from engine import NexusEngine

st.set_page_config(page_title="RiddleNexus Engine", layout="wide")

st.title("🧩 RiddleNexus Engine")
st.markdown("Multi-agent iterative reasoning harness for treasure hunt riddles.")

if "riddle_state" not in st.session_state:
    st.session_state.riddle_state = RiddleState(description="")
if "engine" not in st.session_state:
    st.session_state.engine = NexusEngine(st.session_state.riddle_state)

st.sidebar.header("Input Multi-Modal Clues")
riddle_text = st.sidebar.text_area("Riddle Description / Text Clues", height=150)
if st.sidebar.button("Set Riddle Text"):
    st.session_state.riddle_state.description = riddle_text
    st.sidebar.success("Riddle text updated.")

uploaded_files = st.sidebar.file_uploader("Upload Image Clues (Maps, Ciphers)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
text_files = st.sidebar.file_uploader("Upload Text Clues (.txt, .md)", accept_multiple_files=True, type=['txt', 'md'])

if st.sidebar.button("Add Text Files to Context"):
    if text_files:
        for f in text_files:
            content = f.read().decode("utf-8")
            st.session_state.riddle_state.description += f"\n\n--- Content of {f.name} ---\n{content}"
        st.sidebar.success("Text files appended to Riddle Text.")

if st.sidebar.button("Analyze Images with Gemini"):
    if uploaded_files:
        with st.spinner("Gemini is analyzing images..."):
            for file in uploaded_files:
                mime_type = file.type
                image_data = file.read()
                # Run async call synchronously for Streamlit
                st.session_state.engine.raw_images.append((image_data, mime_type))
                result = asyncio.run(st.session_state.engine.analyze_image_with_gemini(image_data, mime_type))
                st.session_state.riddle_state.image_descriptions.append(f"File {file.name}: {result}")
        st.sidebar.success("Images analyzed.")
    else:
        st.sidebar.warning("No images uploaded.")

# Main Display
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Nexus Reasoning Tree")
    st.write(f"**Current Generation:** {st.session_state.riddle_state.current_generation}")

    num_pistes = st.number_input("Number of Pistes per Cycle", min_value=1, max_value=20, value=5)

    if st.button("▶️ Run Automatic Generation Cycle"):
        with st.status("Running Nexus Cycle...", expanded=True) as status:
            def update_status(msg):
                status.write(msg)
            asyncio.run(st.session_state.engine.run_auto_cycle(n=num_pistes, log_callback=update_status))
            status.update(label="Cycle Complete!", state="complete", expanded=False)
        st.rerun()

    if st.button("🏁 Branch & Iterate Next Loop"):
        with st.spinner("Selecting best scored piste to branch..."):
            st.session_state.engine.branch_next_generation()
        st.success("Branched! Ready for next generation.")
        st.rerun()

    st.markdown("---")
    st.subheader("S.M.A.R.E. Architecture Flow")
    st.info("🔄 Superviseur ➔ 💡 Générateur d'Idées ➔ 🗺️ Cartographe (Deduplication) ➔ ⚙️ Solver de Piste (L'Exécuteur) ➔ ⚖️ Avocat du Diable (Critique) ➔ 🏆 Arbitre (Evaluation)")

    st.markdown("---")
    st.subheader("Visual Reasoning Tree")

    if st.session_state.riddle_state.pistes:
        graph = graphviz.Digraph(engine='dot')
        graph.attr(rankdir='TB')
        for piste_id, piste in st.session_state.riddle_state.pistes.items():
            node_label = f"{piste.id_piste}\\nGen: {piste.generation_depth}\\nScore: {piste.score_elo:.1f}\\nStatus: {piste.statut}"
            color = "white"
            if piste.statut == "Fausse Piste":
                color = "lightpink"
            elif piste.statut == "Bloquée par Parent":
                color = "lightgrey"
            elif piste.id_piste == st.session_state.riddle_state.top_piste_id:
                color = "lightgreen"
            elif piste.statut == "Active":
                color = "lightblue"

            graph.node(piste.id_piste, label=node_label, style='filled', fillcolor=color, shape='box')

            if piste.pistes_parentes:
                for parent_id in piste.pistes_parentes:
                    if parent_id in st.session_state.riddle_state.pistes:
                        graph.edge(parent_id, piste.id_piste)

        st.graphviz_chart(graph)
    else:
        st.info("No pistes generated yet.")

    st.markdown("---")
    st.subheader("Current Pistes Leaderboard")
    current_pistes = st.session_state.riddle_state.get_generation_pistes(st.session_state.riddle_state.current_generation)
    current_pistes.sort(key=lambda x: x.score_elo, reverse=True)

    for idx, piste in enumerate(current_pistes):
        # 1. Status Badges
        status_icon = "⚪"
        if piste.statut == "Active": status_icon = "🟡"
        elif piste.statut == "Fausse Piste": status_icon = "🔴"
        elif piste.statut == "Validée": status_icon = "🟢"
        elif piste.statut == "En attente": status_icon = "🟢"

        expander_title = f"{status_icon} Piste #{idx+1} | Score: {piste.score_elo:.0f} | {piste.id_piste}"

        with st.expander(expander_title, expanded=(idx==0)):
            # 2. Genealogy / Parents
            if piste.pistes_parentes:
                st.caption(f"🧬 **Parents :** {' × '.join(piste.pistes_parentes)}")
            else:
                st.caption("🧬 **Parents :** Nouvelle génération (Racine)")

            # 3. Énigme Réfractée (Simple Output)
            if getattr(piste, "output_simple", None):
                st.info(f"**✨ Énigme Réfractée (Output Simple) :**\n\n{piste.output_simple}")

            tab1, tab2, tab3, tab4 = st.tabs(["💡 Logique", "⚙️ Exécution", "⚖️ Critique & Scores", "📄 JSON"])

            with tab1:
                st.write(f"**Hypothèse :** {piste.hypothese_de_depart}")

            with tab2:
                if piste.protocole_de_test and piste.protocole_de_test != "Aucun":
                    st.write("Vérification de l'hypothèse via script Python...")
                    if piste.resultat_du_test:
                        if "Error" in piste.resultat_du_test or "disabled" in piste.resultat_du_test:
                            st.warning(f"➔ {piste.resultat_du_test.splitlines()[0]}")
                        else:
                            st.success(f"➔ Succès : {piste.resultat_du_test.splitlines()[-1] if piste.resultat_du_test.splitlines() else 'OK'}")

                    with st.expander("Voir le code source et le protocole complet"):
                        st.code(piste.protocole_de_test, language="markdown")
                        if piste.resultat_du_test:
                            st.text(piste.resultat_du_test)
                else:
                    st.write("Aucun protocole d'exécution généré.")

            with tab3:
                if piste.analyse_avocat_du_diable:
                    st.write(f"**Feedback Général :** {piste.analyse_avocat_du_diable.feedback}")
                    if piste.analyse_avocat_du_diable.score_grid:
                        scores = piste.analyse_avocat_du_diable.score_grid

                        def draw_stars(score: int, max_score: int = 10):
                            # score is out of 10. We draw 5 stars.
                            stars_count = round((score / max_score) * 5)
                            return "⭐" * stars_count + "❌" * (5 - stars_count)

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"🔐 Crypto : {draw_stars(scores.cryptography)} ({scores.cryptography}/10)")
                            st.write(f"📜 Histoire : {draw_stars(scores.history)} ({scores.history}/10)")
                        with col_b:
                            st.write(f"🗺️ Géo : {draw_stars(scores.geography)} ({scores.geography}/10)")
                            st.write(f"🧠 Logique : {draw_stars(scores.logic)} ({scores.logic}/10)")
                else:
                    st.write("Pas de critique disponible.")

            with tab4:
                st.json(piste.model_dump())

with col2:
    st.subheader("Context State")
    st.write("**Riddle Text:**")
    st.info(st.session_state.riddle_state.description if st.session_state.riddle_state.description else "No text set.")

    st.write("**Image Extractions:**")
    for desc in st.session_state.riddle_state.image_descriptions:
        st.caption(desc)

    if st.session_state.riddle_state.top_piste_id:
        st.write("**Current Baseline Checkpoint:**")
        st.success(st.session_state.riddle_state.pistes[st.session_state.riddle_state.top_piste_id].hypothese_de_depart)

    st.markdown("---")
    st.subheader("Global State JSON")
    with st.expander("View Global RiddleState JSON"):
        st.json(st.session_state.riddle_state.model_dump(exclude={"pistes"}))
