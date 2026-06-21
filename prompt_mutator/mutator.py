"""
Mutation Engine
Generates adversarial variants of a prompt to stress-test its robustness.
"""

from groq import Groq
import json
import re
import random
import time
from llm_tracer import log_trace

client = Groq()

MUTATION_STRATEGIES = [
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


def generate_llm_mutations(prompt: str, strategies: list[str]) -> dict[str, str]:
    """
    Use Claude to generate semantic mutations of the prompt.
    
    Args:
        prompt: The original prompt to mutate
        strategies: List of strategy names to apply
        
    Returns:
        Dict of {strategy_name: mutated_prompt}
    """
    # Map each strategy name to a clear instruction for the LLM
    strategy_descriptions = {
        "paraphrase": "Rephrase the prompt using different words but keep the same meaning exactly.",
        "tone_casual": "Rewrite the prompt in a very casual, conversational tone (like texting a friend).",
        "tone_formal": "Rewrite the prompt in an overly formal, academic tone.",
        "instruction_reorder": "Reorder the instructions or sentences in the prompt while keeping all content.",
        "ambiguity_injection": "Make one key constraint in the prompt vague or ambiguous.",
        "conflicting_instruction": "Add a subtle instruction that slightly conflicts with the main goal.",
        "over_specification": "Add excessive unnecessary detail and constraints to the prompt.",
        "under_specification": "Remove one important constraint or detail from the prompt.",
        "negation_flip": "Rephrase one instruction using negative framing (e.g. do not X instead of do X).",
    }

    # Filter to only keep strategies the caller asked for
    selected = {k: v for k, v in strategy_descriptions.items() if k in strategies}

    # Tell the LLM to return only JSON, no extra text or markdown
    system_prompt = """You are a prompt mutation engine. Given a prompt and a list of mutation strategies,
generate one mutated version per strategy. Return ONLY a JSON object where keys are strategy names
and values are the mutated prompts. No explanation, no markdown, just raw JSON."""

    user_message = f"""Original prompt:
\"\"\"{prompt}\"\"\"

Apply each of these mutation strategies and return the results as JSON:
{json.dumps(selected, indent=2)}"""

    # Record start time for latency
    start_time = time.time()

    # Make the API call
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
    )

    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000

    raw = response.choices[0].message.content.strip()

    # Log the trace
    log_trace(
        module="mutator",
        prompt=user_message,
        response=raw,
        latency_ms=latency_ms,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        model="llama-3.3-70b-versatile",
    )

    # Strip markdown code fences if present
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


def generate_rule_based_mutations(prompt: str) -> dict[str, str]:
    """
    Generate typo mutations using simple rule-based text manipulation.
    No API call needed for this one.
    
    Args:
        prompt: The original prompt to mutate
        
    Returns:
        Dict of {strategy_name: mutated_prompt}
    """
    mutations = {}

    # Split prompt into individual words
    words = prompt.split()

    # Only apply typo if prompt has more than 3 words
    if len(words) > 3:

        # Pick a random word from the middle of the prompt
        idx = random.randint(1, len(words) - 2)
        word = words[idx]

        # Only mutate words longer than 3 characters
        if len(word) > 3:

            # Pick a random position inside the word and swap two adjacent characters
            i = random.randint(1, len(word) - 2)
            typo_word = word[:i] + word[i+1] + word[i] + word[i+2:]
            words[idx] = typo_word

        # Join words back into a string
        mutations["typo_noise"] = " ".join(words)

    return mutations


def generate_mutations(prompt: str, strategies: list[str] = None) -> dict[str, str]:
    """
    Main function that generates all mutations for a given prompt.
    Coordinates between LLM-based and rule-based mutations.
    
    Args:
        prompt: The original prompt to mutate
        strategies: List of mutation strategies to apply (defaults to all)
        
    Returns:
        Dict of {strategy_name: mutated_prompt}
    """
    # Use all strategies if none are specified
    if strategies is None:
        strategies = MUTATION_STRATEGIES

    # Separate strategies into LLM-based and rule-based
    llm_strategies = [s for s in strategies if s != "typo_noise"]
    rule_strategies = [s for s in strategies if s == "typo_noise"]

    mutations = {}

    # Run LLM mutations if any are requested
    if llm_strategies:
        mutations.update(generate_llm_mutations(prompt, llm_strategies))

    # Run rule-based mutations if typo_noise is requested
    if rule_strategies:
        mutations.update(generate_rule_based_mutations(prompt))

    # Always include the original prompt as a baseline for comparison
    mutations["original"] = prompt

    return mutations