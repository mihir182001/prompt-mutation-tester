from groq import Groq
import json
import re
import time
from llm_tracer import log_trace

client = Groq()


def judge_output(
    original_prompt: str,
    expected_behaviour: str,
    mutated_prompt: str,
    test_input: str,
    actual_output: str,
) -> dict:
    """
    Judge whether a single output meets the expected behaviour.
    Uses a second LLM call as an impartial judge.
    
    Args:
        original_prompt: The original un-mutated prompt
        expected_behaviour: Description of what a correct output should look like
        mutated_prompt: The mutated version of the prompt that was used
        test_input: The test input that was passed
        actual_output: The output produced by the LLM
        
    Returns:
        Dict with passed, score, reasoning and failure_type
    """
    # Tell the judge exactly what to return and in what format
    system_prompt = """You are an objective judge evaluating whether an AI output meets a specified behaviour.
Return ONLY a JSON object with these fields:
- passed: boolean (true if output satisfies expected behaviour)
- score: float between 0 and 1 (1 = perfect, 0 = complete failure)
- reasoning: one sentence explaining your verdict
- failure_type: one of ["format_failure", "content_failure", "instruction_ignored", "partial_failure", null]

No markdown, no explanation outside the JSON."""

    user_message = f"""Original prompt intent:
{original_prompt}

Expected behaviour:
{expected_behaviour}

Actual prompt used (may be mutated):
{mutated_prompt}

Test input given:
{test_input}

Actual output produced:
{actual_output}

Did the output satisfy the expected behaviour?"""

    # Record start time
    start_time = time.time()

    # Make the API call to the judge
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
    )

    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000

    # Extract response
    raw = response.choices[0].message.content.strip()

    # Log the trace
    log_trace(
        module="judge",
        prompt=user_message,
        response=raw,
        latency_ms=latency_ms,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        model="llama-3.3-70b-versatile",
    )

    # Remove ```json fences if the model adds them
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()

    # Convert JSON string to Python dictionary and return
    return json.loads(raw)


def judge_all_results(
    original_prompt: str,
    expected_behaviour: str,
    run_results: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """
    Judge all run results from runner.py.
    
    Args:
        original_prompt: The original un-mutated prompt
        expected_behaviour: Description of what a correct output should look like
        run_results: Output from run_all_mutations() in runner.py
        
    Returns:
        Same structure as run_results but with judgement added to each result
    """
    judged = {}

    # Loop through every strategy and its results
    for strategy, results in run_results.items():
        judged_results = []

        # Judge each individual result
        for result in results:
            judgement = judge_output(
                original_prompt=original_prompt,
                expected_behaviour=expected_behaviour,
                mutated_prompt=result["prompt_used"],
                test_input=result["input"],
                actual_output=result["output"],
            )

            # Add judgement to the result and store it
            judged_results.append({**result, "judgement": judgement})

        judged[strategy] = judged_results

    return judged