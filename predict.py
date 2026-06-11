"""
predict.py
----------
CLI for the India Tech Jobs Salary Predictor.

Commands:
    python predict.py salary   --role "Data Analyst" --city "Bangalore" --exp 3
    python predict.py scrape
    python predict.py pipeline --csv data/india_jobs.csv
    python predict.py train
    python predict.py report
    python predict.py serve
"""

import typer
from rich.console import Console
from rich.table import Table
from rich import box

app = typer.Typer(help="India Tech Jobs — Salary Predictor CLI")
console = Console()


@app.command()
def salary(
    role:    str   = typer.Option(...,  "--role",    "-r", help="Job title, e.g. 'Data Analyst'"),
    city:    str   = typer.Option(...,  "--city",    "-c", help="City, e.g. 'Bangalore'"),
    exp:     float = typer.Option(...,  "--exp",     "-e", help="Years of experience, e.g. 3"),
    skills:  str   = typer.Option("",  "--skills",  "-s", help="Comma-separated skills, e.g. 'Python,SQL'"),
    company: str   = typer.Option("Unknown", "--company", help="Company name (optional)"),
):
    """Predict salary for a given role, city, and experience."""
    from model import predict

    console.print(f"\n[bold cyan]Predicting salary for:[/bold cyan]")
    console.print(f"  Role       : [yellow]{role}[/yellow]")
    console.print(f"  City       : [yellow]{city}[/yellow]")
    console.print(f"  Experience : [yellow]{exp} years[/yellow]")
    if skills:
        console.print(f"  Skills     : [yellow]{skills}[/yellow]")
    if company != "Unknown":
        console.print(f"  Company    : [yellow]{company}[/yellow]")
    console.print()

    try:
        result = predict(job_title=role, city=city, experience_years=exp,
                         skills=skills, company=company)
    except FileNotFoundError:
        console.print("[bold red]Model not found. Run `python predict.py train` first.[/bold red]")
        raise typer.Exit(1)

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Predicted Salary",       f"₹ {result['predicted_salary_lpa']} LPA")
    table.add_row("Range (10th–90th %ile)", f"₹ {result['range_low_lpa']} – {result['range_high_lpa']} LPA")

    console.print(table)
    console.print()


@app.command()
def scrape(
    pages: int  = typer.Option(3,    "--pages", "-p", help="Pages per keyword (default: 3, up to 50 jobs each)"),
    keywords: str | None = typer.Option(None, "--keywords", "-k", help="Comma-separated keywords, e.g. 'data analyst,python developer'"),
):
    """Scrape fresh India tech job listings via API (Adzuna / JSearch), then run the pipeline."""
    from data_pipeline import run_with_scraper

    kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else None
    console.print(f"\n[bold cyan]Scraping job listings...[/bold cyan]")
    if kw_list:
        console.print(f"  Keywords : [yellow]{', '.join(kw_list)}[/yellow]")
    console.print(f"  Pages    : [yellow]{pages} per keyword[/yellow]\n")

    run_with_scraper(pages=pages, keywords=kw_list)
    console.print("[bold green]Scrape + pipeline complete. Run `train` next.[/bold green]\n")


@app.command()
def pipeline(
    csv: str = typer.Option(..., "--csv", help="Path to existing CSV file"),
):
    """Load and clean an existing CSV, then store it in the SQLite database."""
    from data_pipeline import run
    run(csv_path=csv)
    console.print("[bold green]Pipeline complete.[/bold green]")


@app.command()
def train():
    """Train the salary prediction model on the stored database."""
    from model import train as _train, save_model
    model, encoders, top_skills, feature_cols, metrics = _train()
    save_model(model, encoders, top_skills, feature_cols, metrics)
    console.print("\n[bold green]Training complete.[/bold green]")
    console.print(f"  Best model : {metrics['best_model']}")
    console.print(f"  Test MAE   : ₹{metrics['test_mae_lpa']} LPA")
    console.print(f"  R²         : {metrics['test_r2']}\n")
    console.print(f"  Samples    : {metrics['n_samples']}")
    console.print("[dim]Feature importance saved → models/feature_importance.json[/dim]\n")


@app.command()
def report():
    """Generate the HTML dashboard report from stored data."""
    from report import generate
    path = generate()
    console.print(f"[bold green]Report generated:[/bold green] {path}")


@app.command()
def serve(
    port: int  = typer.Option(8080, "--port", "-p", help="Port to listen on (default: 8080)"),
    host: str  = typer.Option("127.0.0.1", "--host", help="Host address (default: 127.0.0.1)"),
):
    """Start the local web dashboard (http://127.0.0.1:8080 by default)."""
    import subprocess, sys
    console.print(f"\n[bold cyan]Starting dashboard on http://{host}:{port}[/bold cyan]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")
    subprocess.run(
        [sys.executable, "server.py", "--host", host, "--port", str(port)],
        check=False,
    )


if __name__ == "__main__":
    app()
