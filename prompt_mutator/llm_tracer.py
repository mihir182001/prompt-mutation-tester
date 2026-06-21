from datetime import datetime
import time
import json

# In-memory store for traces (saved to DB later)
_traces = []


def log_trace(
    module: str,
    prompt: str,
    response: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> dict:
    """
    Log a single LLM API call with full details.
    
    Args:
        module: Which module made the call (mutator, judge, fixer etc.)
        prompt: The prompt that was sent
        response: The response that came back
        latency_ms: How long the call took in milliseconds
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
        model: Which model was used
        
    Returns:
        The trace dict that was saved
    """
    # Estimate cost based on Groq free tier pricing
    cost_estimate = round((input_tokens * 0.0000003) + (output_tokens * 0.0000006), 6)

    trace = {
        "id": len(_traces) + 1,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "module": module,
        "model": model,
        "prompt_preview": prompt[:150] + "..." if len(prompt) > 150 else prompt,
        "response_preview": response[:150] + "..." if len(response) > 150 else response,
        "latency_ms": round(latency_ms),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_estimate": cost_estimate,
    }

    _traces.append(trace)
    return trace


def get_traces(limit: int = 50) -> list:
    """Get most recent traces."""
    return list(reversed(_traces[-limit:]))


def get_trace_stats() -> dict:
    """Get aggregate stats across all traces."""
    if not _traces:
        return {
            "total_calls": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0,
            "total_cost": 0,
            "calls_by_module": {},
        }

    total_tokens = sum(t["total_tokens"] for t in _traces)
    avg_latency = sum(t["latency_ms"] for t in _traces) / len(_traces)
    total_cost = sum(t["cost_estimate"] for t in _traces)

    # Count calls per module
    calls_by_module = {}
    for t in _traces:
        calls_by_module[t["module"]] = calls_by_module.get(t["module"], 0) + 1

    return {
        "total_calls": len(_traces),
        "total_tokens": total_tokens,
        "avg_latency_ms": round(avg_latency),
        "total_cost": round(total_cost, 4),
        "calls_by_module": calls_by_module,
    }


def clear_traces():
    """Clear all traces."""
    global _traces
    _traces = []