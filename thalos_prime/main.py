from __future__ import annotations

import code
import json
import logging
import sys

import click

logger = logging.getLogger(__name__)


@click.group()
@click.option("--log-level", default=None, help="Log level (DEBUG, INFO, WARNING, ERROR)")
def cli(log_level):
    """Thalos Prime — Autonomous Epistemic Compiler"""
    if log_level:
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    else:
        logging.basicConfig(level=logging.WARNING)


@cli.command()
@click.option("--url", default=None, help="URL to ingest")
@click.option("--file", "file_path", default=None, help="File path to ingest")
@click.option("--feed", default=None, help="RSS/Atom/JSON feed URL to ingest")
def ingest(url, file_path, feed):
    """Ingest content from URL, file, or feed."""
    from thalos_prime.core import ThalosCore
    core = ThalosCore()

    if url:
        result = core.ingest_url(url)
        click.echo(json.dumps({"doc_id": result.doc_id, "status": result.status, "word_count": result.word_count, "message": result.message}, indent=2))
    elif file_path:
        result = core.ingest_file(file_path)
        click.echo(json.dumps({"doc_id": result.doc_id, "status": result.status, "word_count": result.word_count, "message": result.message}, indent=2))
    elif feed:
        results = core.ingest_feed(feed)
        output = [{"doc_id": r.doc_id, "status": r.status, "word_count": r.word_count} for r in results]
        click.echo(json.dumps(output, indent=2))
    else:
        click.echo("Provide --url, --file, or --feed", err=True)
        sys.exit(1)


@cli.command()
@click.option("--doc-id", required=True, help="Document ID to run pipeline on")
def pipeline(doc_id):
    """Run the full pipeline on an already-ingested document."""
    from thalos_prime.core import ThalosCore
    core = ThalosCore()
    result = core.run_pipeline(doc_id)
    click.echo(json.dumps(result, indent=2))


@cli.command()
@click.argument("text")
@click.option("--top-k", default=10, help="Number of top results to return")
@click.option("--lang", default=None, help="Language filter (not implemented yet)")
def query(text, top_k, lang):
    """Query the knowledge base."""
    from thalos_prime.core import ThalosCore
    core = ThalosCore()
    result = core.query(text, top_k=top_k)
    click.echo(json.dumps(result, indent=2, default=str))


@cli.command()
@click.option("--host", default=None, help="Host to bind to")
@click.option("--port", default=None, type=int, help="Port to listen on")
def serve(host, port):
    """Start the Thalos Prime API server."""
    import uvicorn
    from thalos_prime.config import get_settings
    settings = get_settings()
    uvicorn.run(
        "thalos_prime.query.api:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=False,
    )


@cli.command()
@click.option("--suite", default="all", type=click.Choice(["fever", "synthetic", "all"]))
@click.option("--limit", default=10, help="Number of samples per suite")
def eval(suite, limit):
    """Run evaluation suite."""
    from thalos_prime.core import ThalosCore
    from thalos_prime.eval.runner import EvalRunner
    core = ThalosCore()
    runner = EvalRunner(core, core.settings)

    if suite == "synthetic":
        results = {"synthetic": runner.run_synthetic(limit=limit)}
    elif suite == "fever":
        results = {"fever_sample": runner.run_fever_sample(limit=limit)}
    else:
        results = runner.run_all()

    runner.print_results(results)


@cli.command()
def shell():
    """Start an interactive Python shell with core pre-loaded."""
    from thalos_prime.core import ThalosCore
    core = ThalosCore()
    banner = (
        "Thalos Prime Interactive Shell\n"
        "Available: core (ThalosCore)\n"
        "Type 'exit()' or Ctrl-D to quit.\n"
    )
    local_vars = {"core": core}
    code.interact(banner=banner, local=local_vars)


@cli.command()
def status():
    """Print database statistics."""
    from sqlalchemy import select, func
    from thalos_prime.config import get_settings
    from thalos_prime.storage.db import Claim, RawDocument, NormalizedDocument, get_session, init_db

    settings = get_settings()
    init_db(settings.database_url)
    with get_session(settings.database_url) as session:
        total_docs = session.execute(select(func.count(RawDocument.id))).scalar() or 0
        total_norm = session.execute(select(func.count(NormalizedDocument.id))).scalar() or 0
        total_claims = session.execute(select(func.count(Claim.id))).scalar() or 0

        status_counts = {}
        for status_val in ["draft", "candidate", "verified", "authoritative", "retracted"]:
            count = session.execute(
                select(func.count(Claim.id)).where(Claim.status == status_val)
            ).scalar() or 0
            status_counts[status_val] = count

    output = {
        "raw_documents": total_docs,
        "normalized_documents": total_norm,
        "total_claims": total_claims,
        "claims_by_status": status_counts,
    }
    click.echo(json.dumps(output, indent=2))


if __name__ == "__main__":
    cli()
