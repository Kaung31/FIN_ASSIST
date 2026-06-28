"""Local Arize Phoenix tracing for the FinAssist pipeline.

Phoenix runs locally with no key and no tokens — it captures spans for what the retriever
fetched and what went to the LLM, which is exactly the lens you want during the live shakeout.
Requires the optional [eval] extra:
    pip install '.[eval]'

Usage:
    python -m finassist.eval.phoenix_trace          # launch the UI + instrument, then idle
    python -m finassist.eval.phoenix_trace --demo   # also run the mock pipeline to seed traces
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)

PROJECT = "finassist"


def _require(module: str):
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            f"{module} not installed. Install the eval extra: pip install '.[eval]'"
        ) from exc


def instrument(tracer_provider=None):  # noqa: ANN001
    """Register OTel tracing and instrument the Azure OpenAI calls. Returns the provider."""
    _require("phoenix")
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from phoenix.otel import register

    if tracer_provider is None:
        tracer_provider = register(project_name=PROJECT, batch=False, auto_instrument=False)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    logger.info("Phoenix instrumentation registered (project=%s).", PROJECT)
    return tracer_provider


def launch():
    """Launch the local Phoenix app and instrument the pipeline. Returns the Phoenix session."""
    phoenix = _require("phoenix")
    session = phoenix.launch_app()
    instrument()
    logger.info("Phoenix UI at %s", getattr(session, "url", "http://localhost:6006"))
    return session


def trace_demo(question: str = "What was total revenue this quarter?") -> int:
    """Launch Phoenix, run the mock pipeline under manual spans, and return captured span count."""
    import time

    from opentelemetry import trace

    from finassist import pipeline
    from finassist.llm.client import LLMClient
    from finassist.qa import rag_chat

    session = launch()
    tracer = trace.get_tracer("finassist")
    llm = LLMClient()

    with tracer.start_as_current_span("finassist.ingest"):
        result = pipeline.ingest("sample.pdf", "sample.pdf", llm)
    with tracer.start_as_current_span("finassist.query") as span:
        span.set_attribute("question", question)
        response = rag_chat.answer(
            question,
            llm,
            company=result["company"],
            fiscal_period=result["fiscal_period"],
        )
        span.set_attribute("grounded", response.grounded)
        span.set_attribute("citations", len(response.citations))

    trace.get_tracer_provider().force_flush()
    time.sleep(2)  # let spans flush to the local collector

    count = _span_count()
    logger.info("Captured %s spans in Phoenix project %r (UI: %s)", count, PROJECT, session.url)
    return count


def _span_count() -> int:
    try:
        from phoenix.client import Client

        df = Client().spans.get_spans_dataframe(project_name=PROJECT)
        return len(df)
    except Exception as exc:  # pragma: no cover - best-effort verification
        logger.warning("Could not query Phoenix span count: %s", exc)
        return -1


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Launch local Phoenix tracing.")
    parser.add_argument("--demo", action="store_true", help="run the mock pipeline to seed traces")
    args = parser.parse_args()

    if args.demo:
        count = trace_demo()
        print(f"Seeded traces: {count} spans captured.")
        session = active_session()
    else:
        session = launch()

    url = getattr(session, "url", "http://localhost:6006")
    print(f"Phoenix running at {url}. Ctrl-C to stop.")
    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Stopping Phoenix.")


def active_session():
    import phoenix as px

    return px.active_session()


if __name__ == "__main__":
    main()
