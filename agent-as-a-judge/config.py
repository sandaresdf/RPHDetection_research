import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class JudgingMode(Enum):
    """Different judging modes as per the paper"""
    PAIRWISE = "pairwise"  # Compare two agent responses
    POINTWISE = "pointwise"  # Evaluate single agent response
    REFERENCE_GUIDED = "reference_guided"  # Evaluate against reference

@dataclass
class EvaluationCriteria:
    """Evaluation criteria for agent judgment"""
    name: str
    description: str
    weight: float = 1.0
    scale: tuple = (1, 5)  # Rating scale

@dataclass
class AgentConfig:
    """Configuration for agents being evaluated"""
    agent_id: str
    name: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000

@dataclass
class JudgeConfig:
    """Configuration for the judge agent"""
    model: str = "gpt-4"  # You can use any LLM
    temperature: float = 0.3  # Lower temperature for more consistent judging
    max_tokens: int = 2000
    use_chain_of_thought: bool = True
    criteria: List[EvaluationCriteria] = None

class Config:
    """Main configuration class"""
    
    # API Configuration (modify as needed)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")
    
    # Default Evaluation Criteria
    DEFAULT_CRITERIA = [
        EvaluationCriteria(
            name="correctness",
            description="Accuracy and correctness of the response",
            weight=1.5
        ),
        EvaluationCriteria(
            name="helpfulness",
            description="How helpful and relevant the response is",
            weight=1.2
        ),
        EvaluationCriteria(
            name="safety",
            description="Safety and ethical considerations",
            weight=1.3
        ),
        EvaluationCriteria(
            name="coherence",
            description="Logical flow and coherence of the response",
            weight=1.0
        ),
        EvaluationCriteria(
            name="completeness",
            description="Completeness of addressing the query",
            weight=1.1
        )
    ]
    
    # Judge Configuration
    JUDGE_CONFIG = JudgeConfig(
        model="gpt-4",
        temperature=0.3,
        max_tokens=2000,
        use_chain_of_thought=True,
        criteria=DEFAULT_CRITERIA
    )