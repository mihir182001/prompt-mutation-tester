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

client = Groq(max_retries=0)

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

    selected = {k: v for k, v in strategy_descriptions.items() if k in strategies}

    system_prompt = """You are a prompt mutation engine. Given a prompt and a list of mutation strategies, generate one mutated version per strategy. Return ONLY a valid JSON object. No thinking. No explanation. No markdown. No code blocks. Just the raw JSON object starting with { and ending with }."""

    user_message = f"""Original prompt:
\"\"\"{prompt}\"\"\"

Apply each of these mutation strategies and return the results as JSON:
{json.dumps(selected, indent=2)}"""

    start_time = time.time()

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=2000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
    )

    latency_ms = (time.time() - start_time) * 1000

    raw = response.choices[0].message.content or ""
    print(f"DEBUG raw response: {repr(raw[:200])}")

    log_trace(
        module="mutator",
        prompt=user_message,
        response=raw[:200],
        latency_ms=latency_ms,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        model="openai/gpt-oss-120b",
    )

    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        raw = json_match.group()

    if not raw:
        return {s: prompt for s in strategies}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {s: prompt for s in strategies}


def generate_rule_based_mutations(prompt: str) -> dict[str, str]:
    mutations = {}

    words = prompt.split()

    if len(words) > 3:
        idx = random.randint(1, len(words) - 2)
        word = words[idx]

        if len(word) > 3:
            i = random.randint(1, len(word) - 2)
            typo_word = word[:i] + word[i+1] + word[i] + word[i+2:]
            words[idx] = typo_word

        mutations["typo_noise"] = " ".join(words)

    return mutations


def generate_mutations(prompt: str, strategies: list[str] = None) -> dict[str, str]:
    if strategies is None:
        strategies = MUTATION_STRATEGIES

    llm_strategies = [s for s in strategies if s != "typo_noise"]
    rule_strategies = [s for s in strategies if s == "typo_noise"]

    mutations = {}

    if llm_strategies:
        mutations.update(generate_llm_mutations(prompt, llm_strategies))

    if rule_strategies:
        mutations.update(generate_rule_based_mutations(prompt))

    mutations["original"] = prompt

    return mutations