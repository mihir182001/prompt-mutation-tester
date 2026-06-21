from groq import Groq
import json
import re
import time
from llm_tracer import log_trace

# Create one shared connection to Groq API
client = Groq()


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    text = ""

    # Extract text from each page
    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text.strip()


def chunk_document(text: str, chunk_size: int = 500) -> list[str]:
    """
    Split a document into chunks for RAG testing.
    
    Args:
        text: The full document text
        chunk_size: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # If adding this paragraph exceeds chunk size, save current chunk
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph

    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def test_rag_prompt(
    prompt: str,
    expected_behaviour: str,
    document_text: str,
    strategies: list[str] = None,
) -> dict:
    """
    Test a RAG extraction prompt against a document across all mutations.
    
    Args:
        prompt: The prompt for extracting information from the document
        expected_behaviour: What a correct extraction should look like
        document_text: The document text to extract from
        strategies: List of mutation strategies to apply
        
    Returns:
        Dict with scores, failures and fixed prompt
    """
    # Import here to avoid circular imports
    from mutator import generate_mutations
    from runner import run_all_mutations
    from judge import judge_all_results
    from reporter import aggregate_scores
    from fixer import suggest_fix

    # Chunk the document
    chunks = chunk_document(document_text)

    # Use first 3 chunks as test inputs
    test_inputs = chunks[:3] if len(chunks) >= 3 else chunks

    print(f"\n📄 Document chunked into {len(chunks)} pieces")
    print(f"Testing against first {len(test_inputs)} chunks\n")

    # Generate mutations of the prompt
    mutations = generate_mutations(prompt, strategies)

    # Run mutations against document chunks
    results = run_all_mutations(mutations, test_inputs)

    # Judge results
    judged = judge_all_results(prompt, expected_behaviour, results)
    report = aggregate_scores(judged)

    # Generate fix if score is below 80
    fixed_prompt = None
    if report["overall_score"] < 80:
        fixed_prompt = suggest_fix(prompt, expected_behaviour, judged)

    # Build strategy scores
    strategy_scores = {
        strategy: {
            "pass_rate": round(s["pass_rate"] * 100),
            "avg_score": round(s["avg_score"] * 100),
            "failure_types": s["failure_types"],
            "passed": s["pass_rate"] == 1.0,
        }
        for strategy, s in report["strategy_scores"].items()
    }

    return {
        "overall_score": report["overall_score"],
        "strategy_scores": strategy_scores,
        "fixed_prompt": fixed_prompt,
        "chunks_tested": len(test_inputs),
        "total_chunks": len(chunks),
    }