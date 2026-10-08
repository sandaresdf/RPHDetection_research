# evaluation_framework.py - Main framework orchestrating evaluations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime

from agent import Agent
from agent import AgentResponse
from judge_agent import JudgeAgent
from judge_agent import JudgmentResult
from judge_agent import PairwiseJudgment
from config import JudgingMode
from config import JudgeConfig

@dataclass
class EvaluationTask:
    """Represents an evaluation task"""
    task_id: str
    query: str
    agents: List['Agent']
    reference_answer: Optional[str] = None
    context: Optional[Dict] = None
    metadata: Dict[str, Any] = None

@dataclass
class EvaluationReport:
    """Comprehensive evaluation report"""
    task_id: str
    query: str
    timestamp: datetime
    judgments: List['JudgmentResult']
    agent_responses: Dict[str, 'AgentResponse']
    rankings: List[tuple]  # [(agent_id, score), ...]
    summary: str
    
    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "judgments": [j.to_dict() for j in self.judgments],
            "agent_responses": {
                k: {
                    "response": v.response,
                    "agent_id": v.agent_id,
                    "response_id": v.response_id
                } for k, v in self.agent_responses.items()
            },
            "rankings": self.rankings,
            "summary": self.summary
        }
    
    def save_to_file(self, filepath: str):
        """Save report to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

class AgentEvaluationFramework:
    """Main framework for Agent-as-a-Judge evaluation"""
    
    def __init__(self, judge_agent: 'JudgeAgent'):
        self.judge = judge_agent
        self.evaluation_history = []
    
    def evaluate_task(self, task: EvaluationTask,
                     mode: 'JudgingMode' = JudgingMode.POINTWISE) -> EvaluationReport:
        """
        Evaluate a task with multiple agents
        
        Args:
            task: EvaluationTask containing query and agents
            mode: Judging mode (pointwise or pairwise)
        
        Returns:
            EvaluationReport with comprehensive results
        """
        import uuid
        
        print(f"\n{'='*60}")
        print(f"Starting evaluation for task: {task.task_id}")
        print(f"Query: {task.query}")
        print(f"Mode: {mode.value}")
        print(f"{'='*60}\n")
        
        # Step 1: Collect responses from all agents
        agent_responses = {}
        for agent in task.agents:
            print(f"Collecting response from agent: {agent.config.agent_id}")
            response = agent.process_query(task.query, task.context)
            agent_responses[agent.config.agent_id] = response
            print(f"Response received (length: {len(response.response)} chars)\n")
        
        # Step 2: Perform judgments based on mode
        judgments = []
        
        if mode == JudgingMode.POINTWISE:
            # Evaluate each agent response independently
            for agent_id, response in agent_responses.items():
                print(f"Judging response from: {agent_id}")
                judgment = self.judge.evaluate_pointwise(
                    query=task.query,
                    response=response,
                    reference=task.reference_answer
                )
                judgments.append(judgment)
                print(f"Judgment complete. Overall score: {judgment.overall_score:.2f}\n")
        
        elif mode == JudgingMode.PAIRWISE:
            # Compare all pairs of agents
            agent_list = list(agent_responses.items())
            for i in range(len(agent_list)):
                for j in range(i + 1, len(agent_list)):
                    agent_id_a, response_a = agent_list[i]
                    agent_id_b, response_b = agent_list[j]
                    
                    print(f"Comparing: {agent_id_a} vs {agent_id_b}")
                    judgment = self.judge.evaluate_pairwise(
                        query=task.query,
                        response_a=response_a,
                        response_b=response_b
                    )
                    judgments.append(judgment)
                    print(f"Winner: {judgment.winner}\n")
        
        # Step 3: Calculate rankings
        rankings = self._calculate_rankings(agent_responses, judgments, mode)
        
        # Step 4: Generate summary
        summary = self._generate_summary(task, judgments, rankings)
        
        # Create report
        report = EvaluationReport(
            task_id=task.task_id,
            query=task.query,
            timestamp=datetime.now(),
            judgments=judgments,
            agent_responses=agent_responses,
            rankings=rankings,
            summary=summary
        )
        
        self.evaluation_history.append(report)
        
        print(f"\n{'='*60}")
        print("Evaluation Complete!")
        print(f"{'='*60}\n")
        
        return report
    
    def _calculate_rankings(self, agent_responses: Dict, judgments: List,
                           mode: 'JudgingMode') -> List[tuple]:
        """Calculate agent rankings based on judgments"""
        
        if mode == JudgingMode.POINTWISE:
            # Simple ranking by overall score
            rankings = []
            for judgment in judgments:
                agent_id = judgment.metadata.get('agent_id')
                if agent_id:
                    rankings.append((agent_id, judgment.overall_score))
            rankings.sort(key=lambda x: x[1], reverse=True)
            return rankings
        
        elif mode == JudgingMode.PAIRWISE:
            # Elo-like ranking based on pairwise wins
            scores = {agent_id: 0.0 for agent_id in agent_responses.keys()}
            
            for judgment in judgments:
                if isinstance(judgment, PairwiseJudgment):
                    winner = judgment.winner
                    if winner != "tie":
                        scores[winner] += 1.0
                    else:
                        # Tie gives 0.5 to both
                        agent_a = judgment.metadata.get('agent_a_id')
                        agent_b = judgment.metadata.get('agent_b_id')
                        if agent_a and agent_b:
                            scores[agent_a] += 0.5
                            scores[agent_b] += 0.5
            
            rankings = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return rankings
        
        return []
    
    def _generate_summary(self, task: EvaluationTask, judgments: List,
                         rankings: List[tuple]) -> str:
        """Generate human-readable summary"""
        
        summary_parts = []
        summary_parts.append(f"Evaluation Summary for: {task.query}\n")
        summary_parts.append(f"Total Agents Evaluated: {len(task.agents)}")
        summary_parts.append(f"Total Judgments: {len(judgments)}\n")
        
        summary_parts.append("Rankings:")
        for rank, (agent_id, score) in enumerate(rankings, 1):
            summary_parts.append(f"  {rank}. {agent_id}: {score:.2f}")
        
        if rankings:
            best_agent = rankings[0][0]
            summary_parts.append(f"\nBest Performing Agent: {best_agent}")
        
        # Add criterion-specific insights
        if judgments:
            summary_parts.append("\nCriterion Analysis:")
            criterion_scores = {}
            
            for judgment in judgments:
                for criterion, score in judgment.scores.items():
                    clean_criterion = criterion.replace('_a', '').replace('_b', '')
                    if clean_criterion not in criterion_scores:
                        criterion_scores[clean_criterion] = []
                    criterion_scores[clean_criterion].append(score)
            
            for criterion, scores in criterion_scores.items():
                avg_score = sum(scores) / len(scores)
                summary_parts.append(f"  {criterion.title()}: {avg_score:.2f} (avg)")
        
        return "\n".join(summary_parts)
    
    def batch_evaluate(self, tasks: List[EvaluationTask],
                      mode: 'JudgingMode' = JudgingMode.POINTWISE) -> List[EvaluationReport]:
        """Evaluate multiple tasks"""
        reports = []
        for i, task in enumerate(tasks, 1):
            print(f"\n\nProcessing task {i}/{len(tasks)}")
            report = self.evaluate_task(task, mode)
            reports.append(report)
        return reports