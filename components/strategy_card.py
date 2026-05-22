import streamlit as st


def render_strategy_card(info: dict) -> None:
    """Render a strategy info card using st.markdown."""
    name = info.get("name", "")
    description = info.get("description", "")
    tags = info.get("tags", [])

    tag_spans = " ".join(f"`{tag}`" for tag in tags)

    md = f"### {name}\n\n{description}"
    if tag_spans:
        md += f"\n\n{tag_spans}"

    st.markdown(md)
