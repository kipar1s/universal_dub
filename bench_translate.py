"""Benchmark utility to compare Hy-MT translation speed GPU vs CPU."""
import time
from typing import Tuple, Optional

PROMPT = (
    "Translate the following English segment into Russian,"
+ "\n\n"
"\"We are going to explore the mysterious caverns beneath\"\")"

def bench(llm: 'llama_cpp.Llama', n=5) -> None:
-"""  # ← REMOVED docstring (dead code)
