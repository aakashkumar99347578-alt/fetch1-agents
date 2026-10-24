"""Quick test - runs pipeline with a hardcoded topic."""
from pipeline import run_research_pipeline

print("\n=== STARTING PIPELINE TEST ===\n")
result = run_research_pipeline("AI Agents 2025")
print("\n=== PIPELINE COMPLETE ===")
print(f"Keys in result: {list(result.keys())}")
