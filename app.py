import streamlit as st
import asyncio
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

    if st.button("🌱 1. Propose Parallel Hypotheses (DeepSeek)"):
        with st.spinner("Proposer Agents generating ideas..."):
            asyncio.run(st.session_state.engine.propose_paths(n=5))
        st.rerun()

    if st.button("🔍 2. Run Critics (DeepSeek)"):
        with st.spinner("Critic Agents evaluating..."):
            asyncio.run(st.session_state.engine.run_critiques())
        st.rerun()

    if st.button("⚔️ 3. Run Elo Tournament (DeepSeek Judge)"):
        with st.spinner("Judge Agent running pairwise comparisons..."):
            asyncio.run(st.session_state.engine.run_elo_tournament())
        st.rerun()

    if st.button("🏁 4. Branch Next Generation (Checkpoint)"):
        with st.spinner("Selecting top Elo idea to branch..."):
            st.session_state.engine.branch_next_generation()
        st.success("Branched! Ready for next generation.")
        st.rerun()

    st.markdown("---")
    st.subheader("Current Ideas Leaderboard")
    current_ideas = st.session_state.riddle_state.get_generation_ideas(st.session_state.riddle_state.current_generation)
    current_ideas.sort(key=lambda x: x.elo_rating, reverse=True)

    for idx, idea in enumerate(current_ideas):
        with st.expander(f"Idea #{idx+1} - Elo: {idea.elo_rating:.0f}", expanded=(idx==0)):
            st.write(f"**Text:** {idea.text}")
            if idea.critique:
                st.write(f"**Critic Feedback:** {idea.critique.feedback}")

with col2:
    st.subheader("Context State")
    st.write("**Riddle Text:**")
    st.info(st.session_state.riddle_state.description if st.session_state.riddle_state.description else "No text set.")

    st.write("**Image Extractions:**")
    for desc in st.session_state.riddle_state.image_descriptions:
        st.caption(desc)

    if st.session_state.riddle_state.top_idea_id:
        st.write("**Current Baseline Checkpoint:**")
        st.success(st.session_state.riddle_state.ideas[st.session_state.riddle_state.top_idea_id].text)
