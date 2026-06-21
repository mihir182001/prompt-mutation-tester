from groq import Groq
import time
from llm_tracer import log_trace

# Create one shared connection to Groq API
client = Groq()


def run_prompt(prompt: str, test_input: str, model: str = "llama-3.3-70b-versatile") -> str:
    """
    Run a single prompt + test input through the LLM and return the output.
    Automatically traces every API call.
    
    Args:
        prompt: The mutated prompt to test
        test_input: The input text to pass along with the prompt
        model: The LLM model to use
        
    Returns:
        The LLM's response as a string
    """
    # Combine the prompt and test input into one message
    full_message = f"{prompt}\n\n{test_input}" if test_input else prompt

    # Record start time for latency tracking
    start_time = time.time()

    # Send to Groq and get response
    response = client.chat.completions.create(
        model=model,
        max_tokens=1000,
        messages=[{"role": "user", "content": full_message}],
    )

    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000

    # Extract response text
    output = response.choices[0].message.content.strip()

    # Log the trace
    log_trace(
        module="runner",
        prompt=full_message,
        response=output,
        latency_ms=latency_ms,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        model=model,
    )

    return output


def run_all_mutations(
    mutations: dict[str, str],
    test_inputs: list[str],
    model: str = "llama-3.3-70b-versatile",
) -> dict[str, list[dict]]:
    """
    Run all mutated prompts against all test inputs and collect outputs.
    
    Args:
        mutations: Dict of {strategy_name: mutated_prompt} from mutator.py
        test_inputs: List of test inputs to pass with each prompt
        model: The LLM model to use
        
    Returns:
        Dict of {strategy_name: [{input, output, prompt_used}, ...]}
    """
    results = {}

    # Loop through every mutation
    for strategy, mutated_prompt in mutations.items():
        strategy_results = []

        # For each mutation, test it against every test input
        for test_input in test_inputs:
            output = run_prompt(mutated_prompt, test_input, model)
            strategy_results.append({
                "input": test_input,
                "output": output,
                "prompt_used": mutated_prompt,
            })

        results[strategy] = strategy_results

    return results