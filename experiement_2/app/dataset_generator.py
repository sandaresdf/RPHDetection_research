# app/dataset_generator.py
import json
import os
from anthropic import Anthropic
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY")

def generate_prompt_injection_dataset(
    output_file="prompt_injection_dataset.jsonl",
    num_samples=5,
    model="claude-sonnet-4-20250514"
):
    """
    Convert RedQueen jailbreak samples into prompt injection attacks.
    """
    
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Load RedQueen dataset
    print("Loading RedQueen dataset...")
    dataset = load_dataset("YifanJ/Red_Queen", split="train", streaming=True)

    results = []
    processed = 0
    
    print(f"Generating {num_samples} prompt injection samples...")
    
    with open(output_file, 'w') as f:
        for row in tqdm(dataset, total=num_samples):
            if processed >= num_samples:
                break
                
            try:
                # Extract the jailbreak content
                query = row.get("query", [])
                if not query:
                    continue
                
                # Get the user's jailbreak attempt
                jailbreak_content = ""
                for msg in query:
                    if msg.get("role") == "user":
                        jailbreak_content = msg.get("content", "")
                        break
                
                if not jailbreak_content:
                    continue

                # read the prompt from the file
                with open(Path("prompts") / "conversion_prompt.txt", "r") as prompt_file:
                    conversion_prompt = prompt_file.read()
                
                # Generate prompt injection using Claude
                response = client.messages.create(
                    model=model,
                    max_tokens=2000,
                    messages=[{
                        "role": "user",
                        "content": conversion_prompt.format(jailbreak=jailbreak_content[:500])
                    }]
                )
                
                # Parse the response
                content = response.content[0].text
                
                # Extract JSON (handle potential markdown code blocks)
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                attack_data = json.loads(content)
                
                # Add metadata
                attack_data["original_jailbreak"] = jailbreak_content[:200]
                attack_data["source"] = "RedQueen"
                attack_data["id"] = processed
                
                # Write to JSONL file
                f.write(json.dumps(attack_data) + "\n")
                results.append(attack_data)
                processed += 1
                
            except Exception as e:
                print(f"\nError processing sample: {e}")
                continue
    
    print(f"\n✓ Generated {processed} prompt injection samples")
    print(f"✓ Saved to {output_file}")
    
    return results


def load_prompt_injection_dataset(file_path="prompt_injection_dataset.jsonl"):
    """Load the generated dataset."""
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples