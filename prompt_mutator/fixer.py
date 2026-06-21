from groq import Groq
import json
import re

# Create one shared connection to Groq API
client = Groq()


def suggest_fix(
    original_prompt: str,
    expected_behaviour: str,
    judged_results: dict[str, list[dict]],
) -> str:
    """
    Analyse failures and suggest an improved version of the prompt.
    
    Args:
        original_prompt: The original prompt that was tested
        expected_behaviour: What the prompt should do
        judged_results: Output from judge_all_results() in judge.py
        
    Returns:
        An improved version of the prompt as a string
    """
    # Collect all failures into a readable summary
    failures = []
    for strategy, results in judged_results.items():
        for result in results:
            if not result["judgement"]["passed"]:
                failures.append({
                    "strategy": strategy,
                    "mutated_prompt": result["prompt_used"],
                    "output": result["output"],
                    "reason": result["judgement"]["reasoning"],
                    "failure_type": result["judgement"]["failure_type"],
                })

    # If no failures found, return original prompt unchanged
    if not failures:
        return original_prompt

    # Build a summary of failures to send to Claude
    failure_summary = json.dumps(failures, indent=2)

    # Tell the LLM exactly what we need
    system_prompt = """You are an expert prompt engineer. 
Your job is to fix a weak prompt based on its failure analysis.
Return ONLY the improved prompt text, nothing else.
No explanation, no preamble, no quotes around it. Just the improved prompt."""

    user_message = f"""Original prompt:
{original_prompt}

Expected behaviour:
{expected_behaviour}

These are the failures found during mutation testing:
{failure_summary}

Write an improved version of the prompt that fixes these weaknesses.
The improved prompt should be robust to paraphrasing, tone changes, and ambiguity."""

    # Send to Groq and get the improved prompt
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
    )

    # Extract and return the improved prompt
    return response.choices[0].message.content.strip()


def verify_fix(
    original_prompt: str,
    fixed_prompt: str,
    expected_behaviour: str,
    test_inputs: list[str],
) -> dict:
    """
    Re-run mutation tests on the fixed prompt and compare scores.
    
    Args:
        original_prompt: The original weak prompt
        fixed_prompt: The improved prompt from suggest_fix()
        expected_behaviour: What the prompt should do
        test_inputs: List of test inputs to use
        
    Returns:
        Dict with original_score, fixed_score and improvement
    """
    # Import here to avoid circular imports
    from mutator import generate_mutations
    from runner import run_all_mutations
    from judge import judge_all_results
    from reporter import aggregate_scores

    # Test the original prompt
    original_mutations = generate_mutations(original_prompt)
    original_results = run_all_mutations(original_mutations, test_inputs)
    original_judged = judge_all_results(original_prompt, expected_behaviour, original_results)
    original_report = aggregate_scores(original_judged)

    # Test the fixed prompt
    fixed_mutations = generate_mutations(fixed_prompt)
    fixed_results = run_all_mutations(fixed_mutations, test_inputs)
    fixed_judged = judge_all_results(fixed_prompt, expected_behaviour, fixed_results)
    fixed_report = aggregate_scores(fixed_judged)

    # Calculate improvement
    original_score = original_report["overall_score"]
    fixed_score = fixed_report["overall_score"]
    improvement = round(fixed_score - original_score, 1)

    return {
        "original_score": original_score,
        "fixed_score": fixed_score,
        "improvement": improvement,
        "original_prompt": original_prompt,
        "fixed_prompt": fixed_prompt,
    }