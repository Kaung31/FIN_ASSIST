"""Thin Streamlit demo. It only calls the FastAPI backend — no business logic here.

Run: streamlit run ui/streamlit_app.py  (with the API running on :8000)
Shows API errors instead of crashing, so problems are visible.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API = os.environ.get("FINASSIST_API", "http://localhost:8000")

st.set_page_config(page_title="FinAssist", layout="wide")
st.title("FinAssist — Earnings Report Assistant")
st.caption("Document analysis, not investment advice.")


def call(method: str, path: str, **kwargs):
    """Call the API; on failure show the error in the UI. Returns (ok, data)."""
    try:
        resp = requests.request(method, f"{API}{path}", timeout=300, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API} — is `uvicorn` running?\n\n{exc}")
        return False, None
    if not resp.ok:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text or "(empty body)"
        st.error(f"API error {resp.status_code} on {path}:\n\n{detail}")
        return False, None
    try:
        return True, resp.json()
    except ValueError:
        st.error(f"API returned non-JSON on {path}:\n\n{resp.text[:500]}")
        return False, None


with st.sidebar:
    st.header("1. Ingest")
    uploaded = st.file_uploader("Earnings PDF", type=["pdf"])
    if uploaded and st.button("Ingest"):
        with st.spinner("Parsing + indexing…"):
            ok, data = call(
                "POST",
                "/ingest",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
            )
        if ok:
            st.session_state["ingested"] = data
            st.success(f"Ingested {data.get('chunks')} chunks")
            st.json(data)

    meta = st.session_state.get("ingested", {})
    company = st.text_input("Company filter (optional)", value=meta.get("company") or "")
    fiscal_period = st.text_input(
        "Fiscal period filter (optional)", value=meta.get("fiscal_period") or ""
    )

params = {"company": company or None, "fiscal_period": fiscal_period or None}
tab_summary, tab_anomalies, tab_qa = st.tabs(["Summary", "Anomalies", "Q&A"])

with tab_summary:
    if st.button("Generate summary"):
        with st.spinner("Summarizing…"):
            ok, data = call("GET", "/summarize", params=params)
        if ok:
            st.subheader(data.get("headline", ""))
            if data.get("key_metrics"):
                st.table(
                    [
                        {
                            "Metric": m["name"],
                            "Value": m["value"],
                            "Prior": m["prior_value"],
                            "YoY %": m["yoy_change_pct"],
                        }
                        for m in data["key_metrics"]
                    ]
                )
            else:
                st.info("No financial tables matched — clear the filters or check the PDF.")
            for g in data.get("guidance", []):
                st.markdown(f"- **Guidance:** {g}")

with tab_anomalies:
    if st.button("Detect anomalies"):
        with st.spinner("Detecting…"):
            ok, data = call("GET", "/anomalies", params=params)
        if ok:
            anomalies = data.get("anomalies", [])
            if not anomalies:
                st.info("No anomalies found (needs a table with current + prior columns).")
            for a in anomalies:
                st.markdown(
                    f"**{a['metric']}** — {a['prior_value']} → {a['current_value']} "
                    f"({a['pct_change']}%, {a['severity']})"
                )
                if a.get("explanation"):
                    st.caption(a["explanation"])

with tab_qa:
    question = st.text_input("Ask a question about the filing")
    if question:
        with st.spinner("Thinking…"):
            ok, data = call("POST", "/chat", json={"question": question, **params})
        if ok:
            if data.get("grounded"):
                st.write(data["answer"])
                cites = ", ".join(
                    f"{c.get('section') or 'source'} (p{c['page']})"
                    for c in data.get("citations", [])
                )
                st.caption("Citations: " + cites)
            else:
                st.warning(data.get("answer", "Not found in the documents."))
