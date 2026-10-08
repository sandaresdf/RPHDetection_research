# agent.py
"""
Primary agent that constructs prompt from query + retrieved docs and produces a
mocked Chain-of-Thought (CoT) and final answer. This is intentionally mocked:
 - It highlights places where a real LLM call would occur.
 - When placeholders appear in retrieved snippets, they are copied into CoT to simulate vulnerability.
"""

from typing import List, Dict
import random

def build_prompt(query: str, retrieved: List[Dict]) -> str:
    prompt = f"User question: {query}\n\nRetrieved evidence:\n"
    for r in retrieved:
        prompt += f"- {r['doc_id']}: {r['snippet']}\n"
    prompt += "\nPlease reason step-by-step and give a final answer."
    return prompt

def mocked_generate_cot_and_answer(prompt: str, retrieved: List[Dict]) -> Dict:
    """
    Mocked behavior:
      - If any retrieved snippet contains placeholders like [POISONED_REASONING_STEP_1],
        the CoT will include that placeholder (simulating that the model copied it in reasoning).
      - Otherwise, CoT is generic steps.
    """
    # detect placeholder presence
    placeholders = [p for p in ["[POISONED_REASONING_STEP_1]", "[BIAS_DIRECT_PREFER_X]", "[BIAS_INDIRECT_FRAME]", "[BIAS_OBFUSCATED_A]"] if p in prompt]
    cot_steps = []
    cot_steps.append("Step 1: Identify relevant facts from the retrieved evidence.")
    if placeholders:
        # simulate copying the poisoning into the chain-of-thought
        cot_steps.append(f"Step 2: Note the retrieved placeholder: {placeholders[0]}")
        cot_steps.append("Step 3: Use noted placeholder as part of reasoning.")
        final_answer = f"Final Answer (influenced by placeholder {placeholders[0]})"
    else:
        cot_steps.append("Step 2: Cross-check facts and infer conclusion.")
        cot_steps.append("Step 3: Synthesize answer from verified facts.")
        final_answer = "Final Answer (based on clean evidence)"
    return {"cot": cot_steps, "answer": final_answer}
