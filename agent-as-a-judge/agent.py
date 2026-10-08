from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from config import AgentConfig
from llm_interface import LLMInterface

@dataclass
class AgentResponse:
    """Represents an agent's response"""
    agent_id: str
    query: str
    response: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))

class Agent:
    """Base agent class that can be evaluated"""
    
    def __init__(self, config: 'AgentConfig', llm_interface: 'LLMInterface'):
        self.config = config
        self.llm = llm_interface
        self.response_history = []
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> AgentResponse:
        """Process a query and return response"""
        
        # Build prompt with context if provided
        prompt = self._build_prompt(query, context)
        
        # Generate response
        response_text = self.llm.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Create response object
        response = AgentResponse(
            agent_id=self.config.agent_id,
            query=query,
            response=response_text,
            metadata={
                "model": self.config.model,
                "temperature": self.config.temperature,
                "context": context
            }
        )
        
        self.response_history.append(response)
        return response
    
    def _build_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        """Build prompt for the agent"""
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            return f"Context:\n{context_str}\n\nQuery: {query}\n\nProvide a helpful and accurate response:"
        return f"Query: {query}\n\nProvide a helpful and accurate response:"
    
    def get_response_by_id(self, response_id: str) -> Optional[AgentResponse]:
        """Retrieve a specific response by ID"""
        for response in self.response_history:
            if response.response_id == response_id:
                return response
        return None