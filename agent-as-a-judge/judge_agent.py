from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from config import JudgingMode
from config import JudgeConfig
from agent import AgentResponse
from llm_interface import LLMInterface

@dataclass
class JudgmentResult:
    """Result of agent judgment"""
    judgment_id: str
    mode: 'JudgingMode'
    scores: Dict[str, float]  # Criterion -> score
    overall_score: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "judgment_id": self.judgment_id,
            "mode": self.mode.value if hasattr(self.mode, 'value') else str(self.mode),
            "scores": self.scores,
            "overall_score": self.overall_score,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

# @dataclass
class PairwiseJudgment(JudgmentResult):
    """Judgment result for pairwise comparison"""
    winner: str  # Agent ID of the winner
    comparison_reasoning: str = ""

class JudgeAgent:
    """Agent-as-a-Judge implementation"""
    
    def __init__(self, config: 'JudgeConfig', llm_interface: 'LLMInterface'):
        self.config = config
        self.llm = llm_interface
        self.judgment_history = []
    
    def evaluate_pointwise(self, query: str, response: 'AgentResponse',
                          reference: Optional[str] = None) -> JudgmentResult:
        """
        Pointwise evaluation: Evaluate a single agent response
        """
        import uuid
        
        # Build evaluation prompt
        prompt = self._build_pointwise_prompt(query, response, reference)
        
        # Generate structured judgment
        schema = self._get_judgment_schema()
        judgment_data = self.llm.generate_structured(
            prompt=prompt,
            schema=schema,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Parse and create judgment result
        scores = {}
        reasoning_parts = []
        
        for criterion in self.config.criteria:
            criterion_key = criterion.name
            score_key = f"{criterion_key}_score"
            reason_key = f"{criterion_key}_reasoning"
            
            if score_key in judgment_data:
                scores[criterion.name] = float(judgment_data[score_key])
                if reason_key in judgment_data:
                    reasoning_parts.append(
                        f"{criterion.name.title()}: {judgment_data[reason_key]}"
                    )
        
        # Calculate weighted overall score
        overall_score = self._calculate_weighted_score(scores)
        
        result = JudgmentResult(
            judgment_id=str(uuid.uuid4()),
            mode=JudgingMode.POINTWISE,
            scores=scores,
            overall_score=overall_score,
            reasoning="\n\n".join(reasoning_parts),
            metadata={
                "query": query,
                "agent_id": response.agent_id,
                "response_id": response.response_id,
                "has_reference": reference is not None
            }
        )
        
        self.judgment_history.append(result)
        return result
    
    def evaluate_pairwise(self, query: str, response_a: 'AgentResponse',
                         response_b: 'AgentResponse') -> PairwiseJudgment:
        """
        Pairwise evaluation: Compare two agent responses
        """
        import uuid
        
        # Build pairwise comparison prompt
        prompt = self._build_pairwise_prompt(query, response_a, response_b)
        
        # Generate structured judgment
        schema = self._get_pairwise_schema()
        judgment_data = self.llm.generate_structured(
            prompt=prompt,
            schema=schema,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Parse results
        scores = {}
        reasoning_parts = []
        
        for criterion in self.config.criteria:
            criterion_key = criterion.name
            score_a_key = f"{criterion_key}_score_a"
            score_b_key = f"{criterion_key}_score_b"
            reason_key = f"{criterion_key}_reasoning"
            
            if score_a_key in judgment_data and score_b_key in judgment_data:
                scores[f"{criterion.name}_a"] = float(judgment_data[score_a_key])
                scores[f"{criterion.name}_b"] = float(judgment_data[score_b_key])
                if reason_key in judgment_data:
                    reasoning_parts.append(
                        f"{criterion.name.title()}: {judgment_data[reason_key]}"
                    )
        
        # Determine winner
        score_a = self._calculate_weighted_score(
            {k.replace('_a', ''): v for k, v in scores.items() if k.endswith('_a')}
        )
        score_b = self._calculate_weighted_score(
            {k.replace('_b', ''): v for k, v in scores.items() if k.endswith('_b')}
        )
        
        winner = response_a.agent_id if score_a > score_b else response_b.agent_id
        if abs(score_a - score_b) < 0.1:
            winner = "tie"
        
        result = PairwiseJudgment(
            judgment_id=str(uuid.uuid4()),
            mode=JudgingMode.PAIRWISE,
            scores=scores,
            overall_score=(score_a + score_b) / 2,
            reasoning="\n\n".join(reasoning_parts),
            winner=winner,
            comparison_reasoning=judgment_data.get("overall_comparison", ""),
            metadata={
                "query": query,
                "agent_a_id": response_a.agent_id,
                "agent_b_id": response_b.agent_id,
                "score_a": score_a,
                "score_b": score_b
            }
        )
        
        self.judgment_history.append(result)
        return result
    
    def _build_pointwise_prompt(self, query: str, response: 'AgentResponse',
                               reference: Optional[str] = None) -> str:
        """Build prompt for pointwise evaluation"""
        
        criteria_desc = "\n".join([
            f"- {c.name.title()} ({c.description}): Rate from {c.scale[0]} to {c.scale[1]}"
            for c in self.config.criteria
        ])
        
        prompt = f"""You are an expert judge evaluating AI agent responses. Your task is to evaluate the following response carefully and objectively.

Query: {query}

Agent Response:
{response.response}
"""
        
        if reference:
            prompt += f"\nReference Answer:\n{reference}\n"
        
        prompt += f"""
Evaluation Criteria:
{criteria_desc}

For each criterion:
1. Provide a score from {self.config.criteria[0].scale[0]} to {self.config.criteria[0].scale[1]}
2. Provide clear reasoning for your score

"""
        
        if self.config.use_chain_of_thought:
            prompt += "Think step-by-step and provide detailed reasoning for each criterion.\n"
        
        return prompt
    
    def _build_pairwise_prompt(self, query: str, response_a: 'AgentResponse',
                              response_b: 'AgentResponse') -> str:
        """Build prompt for pairwise comparison"""
        
        criteria_desc = "\n".join([
            f"- {c.name.title()} ({c.description})"
            for c in self.config.criteria
        ])
        
        prompt = f"""You are an expert judge comparing two AI agent responses. Evaluate both responses carefully and objectively.

Query: {query}

Response A (from {response_a.agent_id}):
{response_a.response}

Response B (from {response_b.agent_id}):
{response_b.response}

Evaluation Criteria:
{criteria_desc}

For each criterion:
1. Rate Response A from {self.config.criteria[0].scale[0]} to {self.config.criteria[0].scale[1]}
2. Rate Response B from {self.config.criteria[0].scale[0]} to {self.config.criteria[0].scale[1]}
3. Explain which response is better for this criterion and why

Finally, provide an overall comparison explaining which response is superior overall.
"""
        
        if self.config.use_chain_of_thought:
            prompt += "\nUse step-by-step reasoning in your evaluation.\n"
        
        return prompt
    
    def _get_judgment_schema(self) -> Dict:
        """Get JSON schema for pointwise judgment"""
        schema = {}
        for criterion in self.config.criteria:
            schema[f"{criterion.name}_score"] = "float (1-5)"
            schema[f"{criterion.name}_reasoning"] = "string"
        return schema
    
    def _get_pairwise_schema(self) -> Dict:
        """Get JSON schema for pairwise judgment"""
        schema = {}
        for criterion in self.config.criteria:
            schema[f"{criterion.name}_score_a"] = "float (1-5)"
            schema[f"{criterion.name}_score_b"] = "float (1-5)"
            schema[f"{criterion.name}_reasoning"] = "string"
        schema["overall_comparison"] = "string"
        return schema
    
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted average score"""
        total_weight = 0
        weighted_sum = 0
        
        for criterion in self.config.criteria:
            if criterion.name in scores:
                weighted_sum += scores[criterion.name] * criterion.weight
                total_weight += criterion.weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0