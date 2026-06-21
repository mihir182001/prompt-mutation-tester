from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Create one shared console for all output
console = Console()


def aggregate_scores(judged_results: dict[str, list[dict]]) -> dict:
    """
    Calculate per-strategy and overall robustness scores.
    
    Args:
        judged_results: Output from judge_all_results() in judge.py
        
    Returns:
        Dict with overall_score and per-strategy breakdown
    """
    strategy_scores = {}

    # Loop through each strategy and calculate its scores
    for strategy, results in judged_results.items():
        scores = [r["judgement"]["score"] for r in results]
        passed = [r["judgement"]["passed"] for r in results]
        failure_types = [
            r["judgement"]["failure_type"]
            for r in results
            if not r["judgement"]["passed"]
        ]

        strategy_scores[strategy] = {
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "pass_rate": sum(passed) / len(passed) if passed else 0,
            "failure_types": failure_types,
            "results": results,
        }

    # Calculate overall score across all strategies
    all_scores = [
        r["judgement"]["score"]
        for results in judged_results.values()
        for r in results
    ]
    overall_score = (sum(all_scores) / len(all_scores) * 100) if all_scores else 0

    return {
        "overall_score": round(overall_score, 1),
        "strategy_scores": strategy_scores,
    }


def render_report(
    original_prompt: str,
    expected_behaviour: str,
    judged_results: dict[str, list[dict]],
) -> dict:
    """
    Render a rich terminal report and return the raw report data.
    
    Args:
        original_prompt: The original un-mutated prompt
        expected_behaviour: Description of what a correct output should look like
        judged_results: Output from judge_all_results() in judge.py
        
    Returns:
        The full report as a dictionary
    """
    report = aggregate_scores(judged_results)
    score = report["overall_score"]

    # Pick colour and verdict based on score
    if score >= 80:
        score_colour = "green"
        verdict = "ROBUST"
    elif score >= 55:
        score_colour = "yellow"
        verdict = "MODERATE"
    else:
        score_colour = "red"
        verdict = "FRAGILE"

    # Print the header panel
    console.print()
    console.print(
        Panel.fit(
            f"[bold]Prompt Robustness Score: [{score_colour}]{score}/100[/{score_colour}]  [{score_colour}]{verdict}[/{score_colour}][/bold]",
            title="[bold cyan]Prompt Mutation Test Report[/bold cyan]",
            border_style="cyan",
        )
    )

    # Print original prompt and expected behaviour
    console.print()
    console.print("[bold]Original Prompt:[/bold]", original_prompt)
    console.print("[bold]Expected Behaviour:[/bold]", expected_behaviour)
    console.print()

    # Build the results table
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Mutation Strategy", width=28)
    table.add_column("Pass Rate", justify="center", width=12)
    table.add_column("Avg Score", justify="center", width=12)
    table.add_column("Failure Types", width=30)
    table.add_column("Status", justify="center", width=10)

    failures = []

    # Add a row for each strategy
    for strategy, data in report["strategy_scores"].items():
        pass_rate = f"{data['pass_rate']*100:.0f}%"
        avg_score = f"{data['avg_score']*100:.0f}/100"
        failure_types = ", ".join(set(data["failure_types"])) if data["failure_types"] else "-"

        if data["pass_rate"] == 1.0:
            status = "[green]PASS[/green]"
        elif data["pass_rate"] >= 0.5:
            status = "[yellow]WARN[/yellow]"
        else:
            status = "[red]FAIL[/red]"
            failures.append((strategy, data))

        table.add_row(strategy, pass_rate, avg_score, failure_types, status)

    console.print(table)

    # Print failure details if any
    if failures:
        console.print()
        console.print("[bold red]Failure Details:[/bold red]")
        for strategy, data in failures:
            console.print(f"\n  [bold]{strategy}[/bold]")
            for r in data["results"]:
                if not r["judgement"]["passed"]:
                    console.print(f"    Input: {r['input'][:60]}")
                    console.print(f"    Reason: {r['judgement']['reasoning']}")

    # Print final verdict
    # Print final verdict
    console.print()
    if score >= 80:
        console.print("[green]Your prompt is robust across all tested mutations.[/green]")
    else:
        console.print("[yellow]Your prompt has weaknesses. Generating auto fix...[/yellow]")
        console.print()

        # Generate and show the fixed prompt
        from fixer import suggest_fix
        fixed_prompt = suggest_fix(original_prompt, expected_behaviour, judged_results)

        console.print(
            Panel(
                fixed_prompt,
                title="[bold green]Suggested Fix[/bold green]",
                border_style="green",
            )
        )

    console.print()
    return report