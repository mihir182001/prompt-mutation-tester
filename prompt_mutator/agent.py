from groq import Groq
import json
import time
from llm_tracer import log_trace

# Create one shared connection to Groq API
client = Groq()

# All available mutation strategies the agent can choose from
AVAILABLE_STRATEGIES = [
    "paraphrase",
    "tone_casual",
    "tone_formal",
    "instruction_reorder",
    "ambiguity_injection",
    "typo_noise",
    "conflicting_instruction",
    "over_specification",
    "under_specification",
    "negation_flip",
]


def agent_decide_next_strategy(
    prompt: str,
    expected_behaviour: str,
    history: list[dict],
    available_strategies: list[str],
) -> str:
    """
    The agent reasons about which mutation strategy to try next
    based on what has failed so far.

    Args:
        prompt: The current prompt being tested
        expected_behaviour: What the prompt should do
        history: List of previous test results
        available_strategies: Strategies not yet tried

    Returns:
        The name of the next strategy to try
    """
    # If no history yet just start with paraphrase
    if not history:
        return "paraphrase"

    # Build a summary of what has been tried so far
    tried_summary = []
    for h in history:
        tried_summary.append({
            "strategy": h["strategy"],
            "score": h["score"],
            "passed": h["passed"],
            "failure_type": h.get("failure_type"),
            "reasoning": h.get("reasoning"),
        })

    system_prompt = """You are an autonomous prompt testing agent.
Your job is to decide which mutation strategy to try next based on what has failed so far.
You must return ONLY the name of one strategy from the available list.
No explanation, no markdown, just the strategy name."""

    user_message = f"""Prompt being tested:
{prompt}

Expected behaviour:
{expected_behaviour}

Strategies tried so far:
{json.dumps(tried_summary, indent=2)}

Available strategies not yet tried:
{json.dumps(available_strategies, indent=2)}

Which strategy should be tried next to best expose weaknesses in this prompt?
Return ONLY the strategy name."""

    # Record start time
    start_time = time.time()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=50,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
    )

    latency_ms = (time.time() - start_time) * 1000
    raw = response.choices[0].message.content.strip().lower()

    # Log the trace
    log_trace(
        module="agent",
        prompt=user_message,
        response=raw,
        latency_ms=latency_ms,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        model="llama-3.3-70b-versatile",
    )

    # Make sure the response is a valid strategy
    for strategy in available_strategies:
        if strategy in raw:
            return strategy

    # Default to first available if agent response is unclear
    return available_strategies[0]


def run_autonomous_agent(
    prompt: str,
    expected_behaviour: str,
    test_inputs: list[str],
    target_score: float = 80.0,
    max_steps: int = 10,
) -> dict:
    """
    Run an autonomous agent that intelligently selects mutation strategies
    based on what's failing, rather than trying all strategies blindly.

    Args:
        prompt: The prompt to test
        expected_behaviour: What the prompt should do
        test_inputs: List of test inputs
        target_score: Stop early if score drops below this
        max_steps: Maximum number of mutation steps

    Returns:
        Dict with full agent history, findings and recommendations
    """
    from runner import run_prompt
    from judge import judge_output
    from fixer import suggest_fix

    print(f"\n🤖 Autonomous agent starting...")
    print(f"Target: find weaknesses in the prompt")
    print(f"Max steps: {max_steps}\n")

    history = []
    remaining_strategies = AVAILABLE_STRATEGIES.copy()
    failures = []
    total_score = 0

    for step in range(1, max_steps + 1):

        # Stop if all strategies have been tried
        if not remaining_strategies:
            print("All strategies exhausted.")
            break

        # Agent decides which strategy to try next
        print(f"Step {step} — Agent deciding next strategy...")
        next_strategy = agent_decide_next_strategy(
            prompt=prompt,
            expected_behaviour=expected_behaviour,
            history=history,
            available_strategies=remaining_strategies,
        )

        print(f"Agent chose: {next_strategy}")
        remaining_strategies.remove(next_strategy)

        # Generate the mutation
        from mutator import generate_mutations
        mutations = generate_mutations(prompt, [next_strategy])
        mutated_prompt = mutations.get(next_strategy, prompt)

        # Test against each input
        step_scores = []
        for test_input in test_inputs:
            output = run_prompt(mutated_prompt, test_input)
            judgement = judge_output(
                original_prompt=prompt,
                expected_behaviour=expected_behaviour,
                mutated_prompt=mutated_prompt,
                test_input=test_input,
                actual_output=output,
            )

            step_scores.append(judgement["score"])

            if not judgement["passed"]:
                failures.append({
                    "strategy": next_strategy,
                    "mutated_prompt": mutated_prompt,
                    "test_input": test_input,
                    "output": output,
                    "failure_type": judgement["failure_type"],
                    "reasoning": judgement["reasoning"],
                })

        avg_score = sum(step_scores) / len(step_scores)
        total_score += avg_score
        passed = avg_score >= 0.8

        print(f"Score: {round(avg_score * 100)}/100 — {'PASS' if passed else 'FAIL'}")

        # Save to history
        history.append({
            "step": step,
            "strategy": next_strategy,
            "mutated_prompt": mutated_prompt,
            "score": round(avg_score * 100),
            "passed": passed,
            "failure_type": failures[-1]["failure_type"] if not passed and failures else None,
            "reasoning": failures[-1]["reasoning"] if not passed and failures else None,
        })

    # Calculate overall robustness
    overall_score = round((total_score / len(history)) * 100, 1) if history else 0

    # Generate fix if there are failures
    fixed_prompt = None
    if failures:
        # Build judged results format for suggest_fix
        judged_results = {}
        for h in history:
            judged_results[h["strategy"]] = [{
                "input": test_inputs[0],
                "output": "",
                "prompt_used": h["mutated_prompt"],
                "judgement": {
                    "passed": h["passed"],
                    "score": h["score"] / 100,
                    "reasoning": h.get("reasoning", ""),
                    "failure_type": h.get("failure_type"),
                }
            }]
        fixed_prompt = suggest_fix(prompt, expected_behaviour, judged_results)

    print(f"\n✅ Agent finished. Overall score: {overall_score}/100")
    print(f"Failures found: {len(failures)}")

    return {
        "overall_score": overall_score,
        "steps_taken": len(history),
        "failures_found": len(failures),
        "history": history,
        "failures": failures,
        "fixed_prompt": fixed_prompt,
        "strategies_tried": [h["strategy"] for h in history],
        "strategies_remaining": remaining_strategies,
    }