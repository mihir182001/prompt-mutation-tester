from groq import Groq
import json

# Create one shared connection to Groq API
client = Groq()


def iterative_optimize(
    prompt: str,
    expected_behaviour: str,
    test_inputs: list[str],
    target_score: float = 90.0,
    max_iterations: int = 5,
) -> dict:
    """
    Automatically fix a prompt repeatedly until it scores above target_score.
    
    Args:
        prompt: The original prompt to optimize
        expected_behaviour: What the prompt should do
        test_inputs: List of test inputs to use
        target_score: Stop when score reaches this (default 90)
        max_iterations: Maximum number of fix attempts (default 5)
        
    Returns:
        Dict with best_prompt, best_score, iterations and full history
    """
    # Import here to avoid circular imports
    from mutator import generate_mutations
    from runner import run_all_mutations
    from judge import judge_all_results
    from reporter import aggregate_scores
    from fixer import suggest_fix

    history = []
    current_prompt = prompt
    best_prompt = prompt
    best_score = 0.0

    print(f"\n🧬 Starting iterative optimization (target: {target_score}/100)")
    print(f"Max iterations: {max_iterations}\n")

    for iteration in range(1, max_iterations + 1):
        print(f"--- Iteration {iteration} ---")
        print(f"Testing: {current_prompt[:60]}...")

        # Run full mutation test on current prompt
        mutations = generate_mutations(current_prompt)
        results = run_all_mutations(mutations, test_inputs)
        judged = judge_all_results(current_prompt, expected_behaviour, results)
        report = aggregate_scores(judged)
        score = report["overall_score"]

        print(f"Score: {score}/100")

        # Save this iteration to history
        history.append({
            "iteration": iteration,
            "prompt": current_prompt,
            "score": score,
            "strategy_scores": {
                strategy: {
                    "pass_rate": round(s["pass_rate"] * 100),
                    "avg_score": round(s["avg_score"] * 100),
                    "passed": s["pass_rate"] == 1.0,
                }
                for strategy, s in report["strategy_scores"].items()
            },
        })

        # Track best result so far
        if score > best_score:
            best_score = score
            best_prompt = current_prompt

        # Stop if target reached
        if score >= target_score:
            print(f"\n✅ Target reached at iteration {iteration}!")
            break

        # Stop if on last iteration
        if iteration == max_iterations:
            print(f"\n⚠️ Max iterations reached. Best score: {best_score}/100")
            break

        # Generate improved prompt for next iteration
        print("Generating fix...")
        fixed = suggest_fix(current_prompt, expected_behaviour, judged)
        current_prompt = fixed

    return {
        "original_prompt": prompt,
        "best_prompt": best_prompt,
        "best_score": best_score,
        "target_score": target_score,
        "target_reached": best_score >= target_score,
        "iterations_taken": len(history),
        "history": history,
    }