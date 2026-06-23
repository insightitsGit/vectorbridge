"""
VectorBridge CLI — vectorbridge run / migrate / status
"""

import json
import sys
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()


@click.group()
@click.version_option("0.1.0", prog_name="vectorbridge")
def main():
    """VectorBridge — Universal vector database migration powered by CHORUS Fabric."""
    pass


@main.command()
@click.option("--config", "-c", required=False, help="Path to config JSON file from dashboard")
@click.option("--license", "-l", "license_key", required=False, envvar="VB_LICENSE_KEY", help="License key (agent mode)")
@click.option("--job", "-j", "job_id", required=False, envvar="VB_JOB_ID", help="Job ID to fetch from dashboard")
@click.option("--no-resume", is_flag=True, default=False, help="Start fresh, ignore checkpoint")
@click.option("--report", "-r", "report_path", default=None, help="Save integrity report to this path")
@click.option("--metric-override", is_flag=True, default=False,
              help="Allow migration even when source/target distance metrics differ (DANGEROUS: queries may return wrong results)")
@click.option("--no-semantic-verify", is_flag=True, default=False,
              help="Skip post-migration semantic probe validation")
@click.option("--semantic-probes", default=100, show_default=True,
              help="Number of probe vectors for semantic validation")
def run(config, license_key, job_id, no_resume, report_path,
        metric_override, no_semantic_verify, semantic_probes):
    """Run a migration job from config file or license key."""
    from .bridge import Bridge

    console.print("\n[bold navy]VectorBridge[/bold navy] [dim]— powered by CHORUS Fabric[/dim]\n")

    try:
        if config:
            console.print(f"[dim]Loading config from[/dim] {config}")
            bridge = Bridge.from_config(config)
        elif license_key:
            console.print(f"[dim]Agent mode — fetching config from dashboard...[/dim]")
            bridge = Bridge.from_license(license_key, job_id)
        else:
            console.print("[red]Error:[/red] provide --config or --license")
            sys.exit(1)

        bridge.resume           = not no_resume
        bridge.metric_override  = metric_override
        bridge.semantic_verify  = not no_semantic_verify
        bridge.semantic_probes  = semantic_probes

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Migrating vectors...", total=None)

            def on_progress(offset, total, stats):
                progress.update(task, completed=offset, total=total,
                                description=f"Migrating — {offset:,}/{total:,} vectors")

            bridge_orig_run = bridge.run

            def run_with_progress():
                from .connectors import get_connector, ConnectorConfig
                from .orchestrator import MigrationJob

                src_cfg = ConnectorConfig(**{
                    k: v for k, v in bridge.source_config.items()
                    if k in ConnectorConfig.__dataclass_fields__
                })
                tgt_cfg = ConnectorConfig(**{
                    k: v for k, v in bridge.target_config.items()
                    if k in ConnectorConfig.__dataclass_fields__
                })
                source = get_connector(bridge.source_type, src_cfg)
                target = get_connector(bridge.target_type, tgt_cfg)
                job = MigrationJob(
                    job_id=bridge.job_id,
                    source=source,
                    target=target,
                    mode=bridge.mode,
                    batch_size=bridge.batch_size,
                    resume=bridge.resume,
                    on_progress=on_progress,
                )
                return job.run()

            report = run_with_progress()

        console.print(report.summary())

        if report_path:
            report.save(report_path)
            console.print(f"[green]Report saved to[/green] {report_path}")

        if report.failed_watermark > 0:
            console.print(f"[yellow]Warning:[/yellow] {report.failed_watermark} watermark failures detected")
            sys.exit(2)

    except PermissionError as e:
        console.print(f"[red]License error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@main.command()
@click.argument("source_type")
@click.argument("target_type")
@click.option("--out", "-o", default="vb_config.json", help="Output config file path")
def init(source_type, target_type, out):
    """Generate a starter config file for a migration."""
    template = {
        "job_id": "job_001",
        "license_key": "vb_live_YOUR_KEY_HERE",
        "source": {"type": source_type},
        "target": {"type": target_type},
        "mode": "full",
        "batch_size": 256,
        "resume": True,
    }

    source_fields = {
        "pgvector": {"host": "localhost", "port": 5432, "database": "mydb",
                     "user": "postgres", "password": "", "table": "embeddings"},
        "chromadb": {"chroma_path": "./chroma_data", "collection_name": "embeddings"},
        "faiss": {"faiss_path": "./index.faiss", "faiss_ids_path": "./ids.npy"},
        "weaviate": {"weaviate_url": "http://localhost:8080", "class_name": "Embedding"},
        "qdrant": {"qdrant_url": "http://localhost:6333", "collection": "embeddings"},
        "pinecone": {"api_key": "", "index_name": "my-index", "namespace": ""},
    }
    target_fields = source_fields

    template["source"].update(source_fields.get(source_type, {}))
    template["target"].update(target_fields.get(target_type, {}))

    with open(out, "w") as f:
        json.dump(template, f, indent=2)

    console.print(f"[green]Config written to[/green] {out}")
    console.print(f"[dim]Edit the credentials then run:[/dim] vectorbridge run --config {out}")


@main.command()
@click.option("--config", "-c", required=True, help="Config JSON file")
def status(config):
    """Show checkpoint status for a job."""
    import os
    from .bridge import Bridge
    from .checkpoint import Checkpoint, checkpoint_path

    cfg = json.loads(open(config).read())
    job_id = cfg.get("job_id", "unknown")
    cp_path = checkpoint_path(job_id)

    if not os.path.exists(cp_path):
        console.print(f"[yellow]No checkpoint found for job[/yellow] {job_id}")
        return

    cp = Checkpoint.load(cp_path)
    tbl = Table(title=f"Job Status — {job_id}", show_header=False)
    tbl.add_column("Field", style="bold")
    tbl.add_column("Value")
    tbl.add_row("Source", cp.source)
    tbl.add_row("Target", cp.target)
    tbl.add_row("Mode", cp.mode)
    tbl.add_row("Vectors transferred", f"{cp.vectors_transferred:,}")
    tbl.add_row("Batches completed", str(cp.batches_completed))
    tbl.add_row("Last offset", str(cp.last_offset))
    tbl.add_row("Started at", cp.started_at)
    tbl.add_row("Updated at", cp.updated_at)
    tbl.add_row("Completed", "✓" if cp.completed else "In progress")
    console.print(tbl)
