"""Streamlit app for interactive multi-agent research with live monitoring."""

import streamlit as st
from src.graph import build_research_graph


st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🔬", layout="wide")


def main():
    st.title("🔬 Multi-Agent Research Assistant")
    st.caption("Autonomous research with specialized AI agents: Search → Summarize → Fact-Check → Report")

    # Sidebar
    with st.sidebar:
        st.header("Agent Pipeline")
        st.markdown(
            "1. 🔍 **Researcher** — Web search & data collection\n"
            "2. 📝 **Summarizer** — Distill key findings\n"
            "3. ✅ **Fact-Checker** — Verify claims\n"
            "4. 📄 **Writer** — Produce final report"
        )
        st.divider()
        st.markdown("### Configuration")
        max_iter = st.slider("Max research iterations", 1, 10, 3)

    # Input
    topic = st.text_input("Enter your research topic:", placeholder="e.g., Latest advances in quantum computing")

    if st.button("🚀 Start Research", type="primary") and topic:
        graph = build_research_graph()

        initial_state = {
            "topic": topic,
            "research_notes": [],
            "summary": "",
            "fact_check_results": [],
            "final_report": "",
            "status": "started",
            "iteration": 0,
        }

        # Progress tracking
        progress = st.progress(0, text="Starting research...")
        status_container = st.container()

        col1, col2 = st.columns([2, 1])

        with col1:
            report_placeholder = st.empty()

        with col2:
            agent_log = st.empty()
            log_text = ""

        steps_total = max_iter + 3  # research iterations + summarize + fact_check + write
        current_step = 0

        for step in graph.stream(initial_state):
            node_name = list(step.keys())[0]
            state_update = step[node_name]
            current_step += 1

            pct = min(current_step / steps_total, 1.0)

            if node_name == "research":
                iteration = state_update.get("iteration", 0)
                notes = len(state_update.get("research_notes", []))
                progress.progress(pct, text=f"🔍 Researching (iteration {iteration})...")
                log_text += f"🔍 Research iter {iteration}: {notes} new sources\n"

            elif node_name == "summarize":
                progress.progress(pct, text="📝 Summarizing findings...")
                log_text += "📝 Summary generated\n"

            elif node_name == "fact_check":
                fc_count = len(state_update.get("fact_check_results", []))
                progress.progress(pct, text="✅ Fact-checking claims...")
                log_text += f"✅ Verified {fc_count} claims\n"

            elif node_name == "write":
                progress.progress(1.0, text="📄 Writing final report...")
                log_text += "📄 Report complete\n"
                report_placeholder.markdown(state_update.get("final_report", ""))

            agent_log.text(log_text)

        progress.progress(1.0, text="✅ Research complete!")
        st.success("Research pipeline finished!")

        # Download button
        final_state = graph.invoke(initial_state)
        st.download_button(
            "📥 Download Report",
            data=final_state["final_report"],
            file_name=f"research_report.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
