"""OpenTelemetry spans for each question, so a slow request can be broken down.

One trace per question:

    answer                      mode, k, tickers, hits, citations, abstained
    ├── retrieve                mode, k, split tickers, hits
    │   ├── embed
    │   ├── search              one per company when a question is split (ticker attribute)
    │   └── rerank              candidates in, hits out
    └── generate                gen_ai.* model and token usage; streamed answers add a `first_token` event

The code only uses opentelemetry-api, whose default tracer does nothing, so tracing costs nothing
until it is switched on. TRACING=console prints finished spans to stdout; TRACING=otlp sends them over
OTLP/HTTP to OTEL_EXPORTER_OTLP_ENDPOINT (default http://localhost:4318, the exporter appends
/v1/traces). OTEL_SERVICE_NAME overrides the service name. Both need the `tracing` extra.
"""

from __future__ import annotations

import logging

from opentelemetry import trace

SERVICE_NAME = "filings-rag"
MODES = ("off", "console", "otlp")

tracer = trace.get_tracer("filings_rag")
log = logging.getLogger(__name__)


def make_exporter(mode: str):
    """The span exporter for a TRACING mode, or None for "off"."""
    if mode not in MODES:
        raise ValueError(f"TRACING must be one of {', '.join(MODES)}, not {mode!r}")
    if mode == "off":
        return None
    try:
        if mode == "console":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            return ConsoleSpanExporter()
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter()
    except ImportError as exc:
        raise RuntimeError(f"TRACING={mode} needs the OpenTelemetry SDK: pip install 'filings-rag[tracing]'") from exc


def configure_tracing(mode: str) -> bool:
    """Install a tracer provider that exports spans as `mode` says. Returns whether it installed one.

    Does nothing for "off", or when a provider is already installed (an earlier call, or a wrapper
    such as `opentelemetry-instrument`), since OpenTelemetry allows only one per process.
    """
    exporter = make_exporter(mode)
    if exporter is None:
        return False
    if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        log.info("A tracer provider is already installed; TRACING=%s leaves it in place", mode)
        return False
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    # Resource.create lets OTEL_SERVICE_NAME / OTEL_RESOURCE_ATTRIBUTES override the default name.
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(BatchSpanProcessor(exporter))  # flushed by the provider's atexit shutdown
    trace.set_tracer_provider(provider)
    log.info("Tracing on: exporting spans to %s", mode)
    return True
