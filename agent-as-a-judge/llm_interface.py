from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import json

class LLMInterface(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def generate_structured(self, prompt: str, schema: Dict, **kwargs) -> Dict:
        """Generate structured response"""
        pass

class GroqInterface(LLMInterface):
    """Groq API interface"""
    
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b"):
        self.api_key = api_key
        self.model = model
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
        except ImportError:
            print("Groq package not installed. Install with: pip install groq")
            self.client = None
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                max_tokens: int = 1000, **kwargs) -> str:
        """Generate response from Groq"""
        if not self.client:
            return "Error: Groq client not initialized"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=1,
                reasoning_effort="medium",
                stream=True,
                stop=None
            )

            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_structured(self, prompt: str, schema: Dict, 
                           temperature: float = 0.3, **kwargs) -> Dict:
        """Generate structured JSON response"""
        structured_prompt = f"{prompt}\n\nProvide your response in the following JSON format:\n{json.dumps(schema, indent=2)}"
        response = self.generate(structured_prompt, temperature=temperature, **kwargs)
        
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        return {"raw_response": response, "parsed": False}

class OpenAIInterface(LLMInterface):
    """OpenAI API interface"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            print("OpenAI package not installed. Install with: pip install openai")
            self.client = None
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                max_tokens: int = 1000, **kwargs) -> str:
        """Generate response from OpenAI"""
        if not self.client:
            return "Error: OpenAI client not initialized"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_structured(self, prompt: str, schema: Dict, 
                           temperature: float = 0.3, **kwargs) -> Dict:
        """Generate structured JSON response"""
        structured_prompt = f"{prompt}\n\nProvide your response in the following JSON format:\n{json.dumps(schema, indent=2)}"
        response = self.generate(structured_prompt, temperature=temperature, **kwargs)
        
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        return {"raw_response": response, "parsed": False}

class AnthropicInterface(LLMInterface):
    """Anthropic Claude API interface"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("Anthropic package not installed. Install with: pip install anthropic")
            self.client = None
    
    def generate(self, prompt: str, temperature: float = 0.7,
                max_tokens: int = 1000, **kwargs) -> str:
        """Generate response from Claude"""
        if not self.client:
            return "Error: Anthropic client not initialized"
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_structured(self, prompt: str, schema: Dict,
                           temperature: float = 0.3, **kwargs) -> Dict:
        """Generate structured JSON response"""
        structured_prompt = f"{prompt}\n\nProvide your response in the following JSON format:\n{json.dumps(schema, indent=2)}"
        response = self.generate(structured_prompt, temperature=temperature, **kwargs)
        
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        return {"raw_response": response, "parsed": False}

class MockLLMInterface(LLMInterface):
    """Mock LLM for testing without API calls"""
    
    def __init__(self, model: str = "mock-model"):
        self.model = model
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response"""
        return f"Mock response to: {prompt[:100]}..."
    
    def generate_structured(self, prompt: str, schema: Dict, **kwargs) -> Dict:
        """Generate mock structured response"""
        mock_response = {}
        for key in schema.keys():
            if "score" in key.lower():
                mock_response[key] = 4
            elif "reasoning" in key.lower():
                mock_response[key] = f"Mock reasoning for {key}"
            else:
                mock_response[key] = f"Mock {key}"
        return mock_response