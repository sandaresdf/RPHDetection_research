import os
import uuid
from typing import List

from evaluation_framework import EvaluationTask
from evaluation_framework import AgentEvaluationFramework
from evaluation_framework import EvaluationReport
from llm_interface import MockLLMInterface, LLMInterface, OpenAIInterface, AnthropicInterface, GroqInterface
from config import AgentConfig
from config import EvaluationCriteria
from agent import Agent
from agent import AgentResponse
from judge_agent import JudgeAgent
from judge_agent import JudgmentResult
from judge_agent import PairwiseJudgment
from config import JudgingMode
from config import JudgeConfig
from config import Config

def create_demo_tasks() -> List[EvaluationTask]:
    """Create demonstration evaluation tasks"""
    
    tasks = [
        EvaluationTask(
            task_id=str(uuid.uuid4()),
            query="What is machine learning and how does it work?",
            agents=[],  # Will be populated
            reference_answer="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It works by using algorithms to identify patterns in data and make predictions or decisions based on those patterns.",
            context={"domain": "AI/ML", "difficulty": "beginner"}
        ),
        EvaluationTask(
            task_id=str(uuid.uuid4()),
            query="Explain the concept of blockchain technology and its potential applications beyond cryptocurrency.",
            agents=[],
            reference_answer=None,
            context={"domain": "Technology", "difficulty": "intermediate"}
        ),
        EvaluationTask(
            task_id=str(uuid.uuid4()),
            query="What are the ethical considerations in developing autonomous vehicles?",
            agents=[],
            reference_answer=None,
            context={"domain": "Ethics/Technology", "difficulty": "advanced"}
        )
    ]
    
    return tasks

def run_pointwise_evaluation_demo():
    """Demonstrate pointwise evaluation"""
    
    print("\n" + "="*80)
    print(" AGENT-AS-A-JUDGE FRAMEWORK - POINTWISE EVALUATION DEMO")
    print("="*80 + "\n")
    
    # Initialize LLM interfaces (using mock for demo)
    print("Initializing LLM interfaces...")
    mock_llm = GroqInterface(api_key=os.environ.get("GROQ_API_KEY", ""))
    
    # Create agents to evaluate
    print("Creating test agents...")

    agent_configs = [
        AgentConfig(
            agent_id="agent_gpt",
            name="GPT Agent",
            model="gpt-oss-20b",
            temperature=0.7
        ),
        AgentConfig(
            agent_id="agent_gpt2",
            name="Claude Agent",
            model="gpt-oss-20b",
            temperature=0.7
        ),
        AgentConfig(
            agent_id="agent_conservative",
            name="Conservative Agent",
            model="gpt-oss-20b",
            temperature=0.3
        )
    ]
    
    agents = [Agent(config, mock_llm) for config in agent_configs]
    
    # Create judge agent
    print("Creating judge agent...")
    judge_config = Config.JUDGE_CONFIG
    judge = JudgeAgent(judge_config, mock_llm)
    
    # Create evaluation framework
    print("Initializing evaluation framework...")
    framework = AgentEvaluationFramework(judge)
    
    # Create evaluation tasks
    print("Creating evaluation tasks...\n")
    tasks = create_demo_tasks()
    
    # Add agents to tasks
    for task in tasks:
        task.agents = agents
    
    # Run pointwise evaluation on first task
    print("\n" + "-"*80)
    print(" Running Pointwise Evaluation")
    print("-"*80 + "\n")
    
    report = framework.evaluate_task(tasks[0], mode=JudgingMode.POINTWISE)
    
    # Display results
    print("\n" + "="*80)
    print(" EVALUATION RESULTS")
    print("="*80 + "\n")
    
    print(report.summary)
    
    print("\n" + "-"*80)
    print(" Detailed Judgments")
    print("-"*80 + "\n")
    
    for judgment in report.judgments:
        agent_id = judgment.metadata.get('agent_id', 'Unknown')
        print(f"\nAgent: {agent_id}")
        print(f"Overall Score: {judgment.overall_score:.2f}")
        print("\nCriterion Scores:")
        for criterion, score in judgment.scores.items():
            print(f"  - {criterion.title()}: {score:.2f}")
        print(f"\nReasoning:\n{judgment.reasoning[:200]}...")
        print("-" * 40)
    
    # Save report
    report.save_to_file(f"pointwise_evaluation_{report.task_id}.json")
    print(f"\nReport saved to: pointwise_evaluation_{report.task_id}.json")
    
    return report

def run_pairwise_evaluation_demo():
    """Demonstrate pairwise evaluation"""
    
    print("\n" + "="*80)
    print(" AGENT-AS-A-JUDGE FRAMEWORK - PAIRWISE EVALUATION DEMO")
    print("="*80 + "\n")
    
    # Initialize components
    print("Initializing components...")
    mock_llm = LLMInterface()
    
    agent_configs = [
        AgentConfig(agent_id="agent_a", name="Agent A", model="gpt-4"),
        AgentConfig(agent_id="agent_b", name="Agent B", model="claude-3"),
    ]
    
    agents = [Agent(config, mock_llm) for config in agent_configs]
    
    judge_config = Config.JUDGE_CONFIG
    judge = JudgeAgent(judge_config, mock_llm)
    
    framework = AgentEvaluationFramework(judge)
    
    # Create task
    tasks = create_demo_tasks()
    tasks[0].agents = agents
    
    # Run pairwise evaluation
    print("\n" + "-"*80)
    print(" Running Pairwise Evaluation")
    print("-"*80 + "\n")
    
    report = framework.evaluate_task(tasks[0], mode=JudgingMode.PAIRWISE)
    
    # Display results
    print("\n" + "="*80)
    print(" PAIRWISE COMPARISON RESULTS")
    print("="*80 + "\n")
    
    print(report.summary)
    
    print("\n" + "-"*80)
    print(" Detailed Pairwise Judgments")
    print("-"*80 + "\n")
    
    for judgment in report.judgments:
        if isinstance(judgment, PairwiseJudgment):
            agent_a = judgment.metadata.get('agent_a_id', 'Unknown')
            agent_b = judgment.metadata.get('agent_b_id', 'Unknown')
            print(f"\nComparison: {agent_a} vs {agent_b}")
            print(f"Winner: {judgment.winner}")
            print(f"\nScore Breakdown:")
            for criterion, score in judgment.scores.items():
                print(f"  - {criterion}: {score:.2f}")
            print(f"\nComparison Reasoning:\n{judgment.comparison_reasoning[:200]}...")
            print("-" * 40)
    
    # Save report
    report.save_to_file(f"pairwise_evaluation_{report.task_id}.json")
    print(f"\nReport saved to: pairwise_evaluation_{report.task_id}.json")
    
    return report

def run_batch_evaluation_demo():
    """Demonstrate batch evaluation"""
    
    print("\n" + "="*80)
    print(" AGENT-AS-A-JUDGE FRAMEWORK - BATCH EVALUATION DEMO")
    print("="*80 + "\n")
    
    # Initialize components
    mock_llm = MockLLMInterface()
    
    agent_configs = [
        AgentConfig(agent_id="agent_1", name="Agent 1", model="model-1"),
        AgentConfig(agent_id="agent_2", name="Agent 2", model="model-2"),
        AgentConfig(agent_id="agent_3", name="Agent 3", model="model-3"),
    ]
    
    agents = [Agent(config, mock_llm) for config in agent_configs]
    
    judge = JudgeAgent(Config.JUDGE_CONFIG, mock_llm)
    framework = AgentEvaluationFramework(judge)
    
    # Create multiple tasks
    tasks = create_demo_tasks()
    for task in tasks:
        task.agents = agents
    
    # Run batch evaluation
    print(f"Running batch evaluation on {len(tasks)} tasks...\n")
    reports = framework.batch_evaluate(tasks, mode=JudgingMode.POINTWISE)
    
    # Aggregate results
    print("\n" + "="*80)
    print(" BATCH EVALUATION SUMMARY")
    print("="*80 + "\n")
    
    print(f"Total Tasks Evaluated: {len(reports)}")
    print(f"Total Judgments Made: {sum(len(r.judgments) for r in reports)}")
    
    # Calculate average scores per agent across all tasks
    agent_total_scores = {}
    agent_task_counts = {}
    
    for report in reports:
        for agent_id, score in report.rankings:
            if agent_id not in agent_total_scores:
                agent_total_scores[agent_id] = 0.0
                agent_task_counts[agent_id] = 0
            agent_total_scores[agent_id] += score
            agent_task_counts[agent_id] += 1
    
    print("\nAverage Agent Performance Across All Tasks:")
    for agent_id in sorted(agent_total_scores.keys()):
        avg_score = agent_total_scores[agent_id] / agent_task_counts[agent_id]
        print(f"  {agent_id}: {avg_score:.2f}")
    
    # Save all reports
    for i, report in enumerate(reports, 1):
        filename = f"batch_evaluation_task_{i}_{report.task_id}.json"
        report.save_to_file(filename)
        print(f"\nTask {i} report saved to: {filename}")
    
    return reports

def demonstrate_real_world_scenario():
    """Demonstrate a real-world evaluation scenario"""
    
    print("\n" + "="*80)
    print(" REAL-WORLD SCENARIO: Customer Support Agent Evaluation")
    print("="*80 + "\n")
    
    # Custom criteria for customer support
    customer_support_criteria = [
        EvaluationCriteria(
            name="empathy",
            description="Shows understanding and empathy towards customer concerns",
            weight=1.5,
            scale=(1, 5)
        ),
        EvaluationCriteria(
            name="accuracy",
            description="Provides accurate and correct information",
            weight=1.8,
            scale=(1, 5)
        ),
        EvaluationCriteria(
            name="clarity",
            description="Communicates clearly and avoids jargon",
            weight=1.3,
            scale=(1, 5)
        ),
        EvaluationCriteria(
            name="actionability",
            description="Provides clear next steps and actionable solutions",
            weight=1.6,
            scale=(1, 5)
        ),
        EvaluationCriteria(
            name="professionalism",
            description="Maintains professional tone and courtesy",
            weight=1.2,
            scale=(1, 5)
        )
    ]
    
    # Create custom judge config
    custom_judge_config = JudgeConfig(
        model="gpt-4",
        temperature=0.2,
        max_tokens=2000,
        use_chain_of_thought=True,
        criteria=customer_support_criteria
    )
    
    # Initialize components
    mock_llm = MockLLMInterface()
    
    # Create customer support agents
    support_agents = [
        Agent(
            AgentConfig(
                agent_id="support_agent_friendly",
                name="Friendly Support Agent",
                model="gpt-4",
                temperature=0.8
            ),
            mock_llm
        ),
        Agent(
            AgentConfig(
                agent_id="support_agent_technical",
                name="Technical Support Agent",
                model="gpt-4",
                temperature=0.3
            ),
            mock_llm
        ),
        Agent(
            AgentConfig(
                agent_id="support_agent_balanced",
                name="Balanced Support Agent",
                model="claude-3",
                temperature=0.6
            ),
            mock_llm
        )
    ]
    
    # Create judge with custom criteria
    judge = JudgeAgent(custom_judge_config, mock_llm)
    framework = AgentEvaluationFramework(judge)
    
    # Create customer support scenarios
    support_tasks = [
        EvaluationTask(
            task_id=str(uuid.uuid4()),
            query="I've been waiting for my order for 3 weeks and the tracking shows it's stuck. I'm very frustrated. What can you do to help?",
            agents=support_agents,
            reference_answer=None,
            context={
                "scenario": "delayed_order",
                "customer_emotion": "frustrated",
                "priority": "high"
            }
        ),
        EvaluationTask(
            task_id=str(uuid.uuid4()),
            query="How do I reset my password? I've tried the 'forgot password' link but I'm not receiving any emails.",
            agents=support_agents,
            reference_answer=None,
            context={
                "scenario": "technical_issue",
                "customer_emotion": "confused",
                "priority": "medium"
            }
        )
    ]
    
    # Evaluate
    print("Evaluating customer support agents on real-world scenarios...\n")
    reports = framework.batch_evaluate(support_tasks, mode=JudgingMode.POINTWISE)
    
    # Display results
    print("\n" + "="*80)
    print(" CUSTOMER SUPPORT EVALUATION RESULTS")
    print("="*80 + "\n")
    
    for i, report in enumerate(reports, 1):
        print(f"\nScenario {i}: {report.query[:80]}...")
        print("\nAgent Rankings:")
        for rank, (agent_id, score) in enumerate(report.rankings, 1):
            print(f"  {rank}. {agent_id}: {score:.2f}")
        print("\n" + "-"*80)
    
    return reports

# Additional utility functions

def compare_judging_modes():
    """Compare pointwise vs pairwise judging modes"""
    
    print("\n" + "="*80)
    print(" COMPARISON: Pointwise vs Pairwise Evaluation")
    print("="*80 + "\n")
    
    mock_llm = MockLLMInterface()
    
    agents = [
        Agent(AgentConfig(agent_id=f"agent_{i}", name=f"Agent {i}", model="model"), mock_llm)
        for i in range(1, 4)
    ]
    
    judge = JudgeAgent(Config.JUDGE_CONFIG, mock_llm)
    framework = AgentEvaluationFramework(judge)
    
    task = EvaluationTask(
        task_id=str(uuid.uuid4()),
        query="What are the benefits of renewable energy?",
        agents=agents
    )
    
    # Pointwise evaluation
    print("Running POINTWISE evaluation...")
    pointwise_report = framework.evaluate_task(task, mode=JudgingMode.POINTWISE)
    
    print("\nPointwise Rankings:")
    for rank, (agent_id, score) in enumerate(pointwise_report.rankings, 1):
        print(f"  {rank}. {agent_id}: {score:.2f}")
    
    # Pairwise evaluation
    print("\n" + "-"*80)
    print("\nRunning PAIRWISE evaluation...")
    
    # Reset task agents for fresh evaluation
    task.agents = [
        Agent(AgentConfig(agent_id=f"agent_{i}", name=f"Agent {i}", model="model"), mock_llm)
        for i in range(1, 4)
    ]
    
    pairwise_report = framework.evaluate_task(task, mode=JudgingMode.PAIRWISE)
    
    print("\nPairwise Rankings:")
    for rank, (agent_id, score) in enumerate(pairwise_report.rankings, 1):
        print(f"  {rank}. {agent_id}: {score:.2f}")
    
    print("\n" + "="*80)
    print(" COMPARISON INSIGHTS")
    print("="*80 + "\n")
    
    print("Pointwise Evaluation:")
    print("  - Evaluates each agent independently")
    print("  - Faster for many agents")
    print("  - Provides absolute scores")
    print(f"  - Total judgments: {len(pointwise_report.judgments)}")
    
    print("\nPairwise Evaluation:")
    print("  - Compares agents directly")
    print("  - More nuanced comparisons")
    print("  - Better for relative ranking")
    print(f"  - Total judgments: {len(pairwise_report.judgments)}")
    
    return pointwise_report, pairwise_report

def analyze_judge_consistency():
    """Analyze consistency of judge evaluations"""
    
    print("\n" + "="*80)
    print(" JUDGE CONSISTENCY ANALYSIS")
    print("="*80 + "\n")
    
    mock_llm = MockLLMInterface()
    judge = JudgeAgent(Config.JUDGE_CONFIG, mock_llm)
    
    # Create same agent, same query, evaluate multiple times
    agent = Agent(
        AgentConfig(agent_id="test_agent", name="Test Agent", model="model"),
        mock_llm
    )
    
    query = "Explain quantum computing in simple terms."
    response = agent.process_query(query)
    
    print(f"Query: {query}")
    print(f"Evaluating same response {5} times to check consistency...\n")
    
    judgments = []
    for i in range(5):
        judgment = judge.evaluate_pointwise(query, response)
        judgments.append(judgment)
        print(f"Trial {i+1}: Overall Score = {judgment.overall_score:.2f}")
    
    # Calculate statistics
    scores = [j.overall_score for j in judgments]
    avg_score = sum(scores) / len(scores)
    variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
    std_dev = variance ** 0.5
    
    print(f"\nConsistency Statistics:")
    print(f"  Average Score: {avg_score:.2f}")
    print(f"  Standard Deviation: {std_dev:.2f}")
    print(f"  Min Score: {min(scores):.2f}")
    print(f"  Max Score: {max(scores):.2f}")
    print(f"  Range: {max(scores) - min(scores):.2f}")
    
    if std_dev < 0.3:
        print("\n✓ Judge shows HIGH consistency")
    elif std_dev < 0.6:
        print("\n✓ Judge shows MODERATE consistency")
    else:
        print("\n⚠ Judge shows LOW consistency - consider adjusting temperature")
    
    return judgments

# Main execution function

def main():
    """Main function to run demonstrations"""
    
    print("\n" + "="*80)
    print("  AGENT-AS-A-JUDGE EVALUATION FRAMEWORK")
    print("  Implementation for Research: Secure and Reliable Agentic AI")
    print("="*80 + "\n")
    
    print("This framework implements the Agent-as-a-Judge paradigm for")
    print("evaluating AI agents in real-time with behavioral governance.\n")
    
    print("Available demonstrations:")
    print("  1. Pointwise Evaluation Demo")
    print("  2. Pairwise Evaluation Demo")
    print("  3. Batch Evaluation Demo")
    print("  4. Real-World Customer Support Scenario")
    print("  5. Compare Judging Modes")
    print("  6. Judge Consistency Analysis")
    print("  7. Run All Demos")
    print("  0. Exit")
    
    while True:
        choice = input("\nSelect demo to run (0-7): ").strip()
        
        if choice == '0':
            print("\nExiting... Thank you for using the framework!")
            break
        elif choice == '1':
            run_pointwise_evaluation_demo()
        elif choice == '2':
            run_pairwise_evaluation_demo()
        elif choice == '3':
            run_batch_evaluation_demo()
        elif choice == '4':
            demonstrate_real_world_scenario()
        elif choice == '5':
            compare_judging_modes()
        elif choice == '6':
            analyze_judge_consistency()
        elif choice == '7':
            print("\nRunning all demonstrations...\n")
            run_pointwise_evaluation_demo()
            input("\nPress Enter to continue to next demo...")
            run_pairwise_evaluation_demo()
            input("\nPress Enter to continue to next demo...")
            run_batch_evaluation_demo()
            input("\nPress Enter to continue to next demo...")
            demonstrate_real_world_scenario()
            input("\nPress Enter to continue to next demo...")
            compare_judging_modes()
            input("\nPress Enter to continue to next demo...")
            analyze_judge_consistency()
            print("\nAll demonstrations completed!")
        else:
            print("Invalid choice. Please select 0-7.")
        
        cont = input("\nRun another demo? (y/n): ").strip().lower()
        if cont != 'y':
            print("\nThank you for using the Agent-as-a-Judge framework!")
            break

if __name__ == "__main__":
    main()