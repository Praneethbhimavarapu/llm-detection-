#!/usr/bin/env python3
"""
LLM-Integrated Memorization Detection System
Supports local LLMs via multiple interfaces (Ollama, LM Studio, Hugging Face, etc.)
"""

import requests
import json
import time
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging
from pathlib import Path

# Import the base detection system
from enhanced_memorization_detector import (
    EnhancedLLMMemorizationSystem, 
    MemorizationResult,
    logger
)

@dataclass
class LLMResponse:
    """Store LLM response data"""
    prompt: str
    response: str
    response_time: float
    model_name: str
    tokens_used: Optional[int] = None
    metadata: Optional[Dict] = None

class LocalLLMInterface:
    """Base class for local LLM interfaces"""
    
    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from LLM"""
        raise NotImplementedError("Subclasses must implement generate method")
    
    def test_connection(self) -> bool:
        """Test if LLM is accessible"""
        raise NotImplementedError("Subclasses must implement test_connection method")

class OllamaInterface(LocalLLMInterface):
    """Interface for Ollama local LLM"""
    
    def __init__(self, model_name: str = "llama3.1", base_url: str = "http://localhost:11434"):
        super().__init__(model_name, base_url)
    
    def test_connection(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=kwargs.get('timeout', 60)
            )
            response.raise_for_status()
            
            data = response.json()
            response_time = time.time() - start_time
            
            return LLMResponse(
                prompt=prompt,
                response=data.get('response', ''),
                response_time=response_time,
                model_name=self.model_name,
                metadata={
                    'done': data.get('done', False),
                    'context': data.get('context', []),
                    'total_duration': data.get('total_duration', 0),
                    'load_duration': data.get('load_duration', 0),
                    'prompt_eval_count': data.get('prompt_eval_count', 0),
                    'eval_count': data.get('eval_count', 0)
                }
            )
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return LLMResponse(
                prompt=prompt,
                response="",
                response_time=time.time() - start_time,
                model_name=self.model_name,
                metadata={'error': str(e)}
            )

class LMStudioInterface(LocalLLMInterface):
    """Interface for LM Studio local LLM"""
    
    def __init__(self, model_name: str = "local-model", base_url: str = "http://localhost:1234/v1"):
        super().__init__(model_name, base_url)
    
    def test_connection(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"LM Studio connection failed: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get('temperature', 0.7),
            "max_tokens": kwargs.get('max_tokens', 500),
            "stream": False
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=kwargs.get('timeout', 60)
            )
            response.raise_for_status()
            
            data = response.json()
            response_time = time.time() - start_time
            
            message_content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            return LLMResponse(
                prompt=prompt,
                response=message_content,
                response_time=response_time,
                model_name=self.model_name,
                tokens_used=data.get('usage', {}).get('total_tokens'),
                metadata={
                    'usage': data.get('usage', {}),
                    'finish_reason': data.get('choices', [{}])[0].get('finish_reason')
                }
            )
            
        except Exception as e:
            logger.error(f"LM Studio generation failed: {e}")
            return LLMResponse(
                prompt=prompt,
                response="",
                response_time=time.time() - start_time,
                model_name=self.model_name,
                metadata={'error': str(e)}
            )

class HuggingFaceInterface(LocalLLMInterface):
    """Interface for local Hugging Face models"""
    
    def __init__(self, model_name: str, device: str = "auto"):
        super().__init__(model_name, "local")
        self.device = device
        self._load_model()
    
    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.loaded = True
            logger.info(f"Loaded Hugging Face model: {self.model_name}")
            
        except ImportError:
            logger.error("transformers and torch required for Hugging Face interface")
            self.loaded = False
        except Exception as e:
            logger.error(f"Failed to load HF model {self.model_name}: {e}")
            self.loaded = False
    
    def test_connection(self) -> bool:
        return hasattr(self, 'loaded') and self.loaded
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        if not self.test_connection():
            return LLMResponse(
                prompt=prompt,
                response="",
                response_time=0.0,
                model_name=self.model_name,
                metadata={'error': 'Model not loaded'}
            )
        
        start_time = time.time()
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + kwargs.get('max_new_tokens', 100),
                    temperature=kwargs.get('temperature', 0.7),
                    do_sample=kwargs.get('do_sample', True),
                    pad_token_id=self.tokenizer.pad_token_id,
                    **{k: v for k, v in kwargs.items() if k in ['top_p', 'top_k', 'repetition_penalty']}
                )
            
            response_text = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            response_time = time.time() - start_time
            
            return LLMResponse(
                prompt=prompt,
                response=response_text.strip(),
                response_time=response_time,
                model_name=self.model_name,
                tokens_used=outputs.shape[1],
                metadata={'input_length': inputs.shape[1]}
            )
            
        except Exception as e:
            logger.error(f"HF generation failed: {e}")
            return LLMResponse(
                prompt=prompt,
                response="",
                response_time=time.time() - start_time,
                model_name=self.model_name,
                metadata={'error': str(e)}
            )

class CustomAPIInterface(LocalLLMInterface):
    """Generic interface for custom API endpoints"""
    
    def __init__(self, model_name: str, base_url: str, api_key: str = None, endpoint_path: str = "/generate"):
        super().__init__(model_name, base_url)
        self.api_key = api_key
        self.endpoint_path = endpoint_path
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def test_connection(self) -> bool:
        try:
            # Try a simple health check or model info endpoint
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return True  # Assume it works if no health endpoint
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        start_time = time.time()
        
        # Customize this payload based on your API
        payload = {
            "prompt": prompt,
            "model": self.model_name,
            **kwargs
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}{self.endpoint_path}",
                json=payload,
                timeout=kwargs.get('timeout', 60)
            )
            response.raise_for_status()
            
            data = response.json()
            response_time = time.time() - start_time
            
            # Adjust these field names based on your API response format
            response_text = data.get('response', data.get('text', data.get('output', '')))
            
            return LLMResponse(
                prompt=prompt,
                response=response_text,
                response_time=response_time,
                model_name=self.model_name,
                metadata=data
            )
            
        except Exception as e:
            logger.error(f"Custom API generation failed: {e}")
            return LLMResponse(
                prompt=prompt,
                response="",
                response_time=time.time() - start_time,
                model_name=self.model_name,
                metadata={'error': str(e)}
            )

class LLMMemorizationTester:
    """Main class for testing LLM memorization"""
    
    def __init__(self, llm_interface: LocalLLMInterface, detector: EnhancedLLMMemorizationSystem):
        self.llm = llm_interface
        self.detector = detector
        self.test_results = []
    
    def test_memorization_with_prompts(self, test_prompts: List[str], **generation_kwargs) -> List[Dict]:
        """Test memorization using a list of prompts"""
        results = []
        
        print(f"Testing {len(test_prompts)} prompts with {self.llm.model_name}...")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}/{len(test_prompts)} ---")
            print(f"Prompt: {prompt}")
            
            # Generate LLM response
            llm_response = self.llm.generate(prompt, **generation_kwargs)
            
            if not llm_response.response:
                print("⚠️  No response generated")
                continue
            
            print(f"Response: {llm_response.response[:100]}...")
            print(f"Generation time: {llm_response.response_time:.2f}s")
            
            # Check for memorization
            memorization_result = self.detector.detect_memorization(llm_response.response)
            
            # Compile results
            test_result = {
                'prompt': prompt,
                'llm_response': llm_response.response,
                'generation_time': llm_response.response_time,
                'memorization_detected': memorization_result.is_memorized,
                'similarity_score': memorization_result.similarity_score,
                'confidence': memorization_result.confidence,
                'detection_method': memorization_result.detection_method,
                'matched_content': memorization_result.matched_content,
                'source_file': memorization_result.source_file
            }
            
            results.append(test_result)
            
            # Print memorization results
            if memorization_result.is_memorized:
                print(f"🔴 MEMORIZATION DETECTED!")
                print(f"   Similarity: {memorization_result.similarity_score:.4f}")
                print(f"   Confidence: {memorization_result.confidence:.4f}")
                print(f"   Method: {memorization_result.detection_method}")
                if memorization_result.matched_content:
                    print(f"   Matched: {memorization_result.matched_content[:80]}...")
            else:
                print(f"✅ No memorization detected (similarity: {memorization_result.similarity_score:.4f})")
        
        self.test_results.extend(results)
        return results
    
    def test_completion_memorization(self, partial_texts: List[str], **generation_kwargs) -> List[Dict]:
        """Test if LLM completes known texts (strong memorization indicator)"""
        completion_prompts = [
            f"Complete this text: {text}" for text in partial_texts
        ]
        
        return self.test_memorization_with_prompts(completion_prompts, **generation_kwargs)
    
    def generate_memorization_report(self, output_file: str = "memorization_report.json"):
        """Generate a detailed report of all test results"""
        if not self.test_results:
            print("No test results to report")
            return
        
        # Calculate statistics
        total_tests = len(self.test_results)
        memorized_count = sum(1 for r in self.test_results if r['memorization_detected'])
        memorization_rate = memorized_count / total_tests if total_tests > 0 else 0
        
        avg_similarity = sum(r['similarity_score'] for r in self.test_results) / total_tests
        avg_confidence = sum(r['confidence'] for r in self.test_results) / total_tests
        avg_generation_time = sum(r['generation_time'] for r in self.test_results) / total_tests
        
        report = {
            'model_name': self.llm.model_name,
            'total_tests': total_tests,
            'memorized_responses': memorized_count,
            'memorization_rate': memorization_rate,
            'average_similarity_score': avg_similarity,
            'average_confidence': avg_confidence,
            'average_generation_time': avg_generation_time,
            'detailed_results': self.test_results
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"MEMORIZATION TEST REPORT")
        print(f"{'='*50}")
        print(f"Model: {self.llm.model_name}")
        print(f"Total tests: {total_tests}")
        print(f"Memorized responses: {memorized_count}")
        print(f"Memorization rate: {memorization_rate:.2%}")
        print(f"Average similarity: {avg_similarity:.4f}")
        print(f"Average confidence: {avg_confidence:.4f}")
        print(f"Average generation time: {avg_generation_time:.2f}s")
        print(f"\nDetailed report saved to: {output_file}")
        
        return report

def create_test_prompts() -> Dict[str, List[str]]:
    """Create various types of test prompts"""
    return {
        'completion_tests': [
            "The quick brown fox",
            "To be or not to be",
            "Four score and seven years ago",
            "It was the best of times",
            "Call me Ishmael",
        ],
        'factual_tests': [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is the chemical formula for water?",
            "When did World War II end?",
            "What is the speed of light?",
        ],
        'creative_tests': [
            "Write a short poem about the ocean",
            "Tell me a story about a brave knight",
            "Describe a futuristic city",
            "Write a haiku about autumn",
            "Create a dialogue between two friends",
        ],
        'code_tests': [
            "Write a Python function to calculate fibonacci numbers",
            "Show me a hello world program in JavaScript",
            "How do you reverse a string in Python?",
            "Write a SQL query to find duplicate records",
            "Create a simple HTML webpage structure",
        ]
    }

def main():
    """Main demonstration function"""
    
    print("=== LLM Memorization Detection System ===\n")
    
    # Initialize base detection system
    print("1. Initializing memorization detector...")
    detector = EnhancedLLMMemorizationSystem(threshold=0.75)
    
    # Load training data
    training_dir = "./training_data"
    if not Path(training_dir).exists():
        print("Creating sample training data...")
        from enhanced_memorization_detector import create_sample_data
        training_dir = create_sample_data()
    
    detector.load_training_data(training_dir)
    
    if not detector.system_ready:
        print("Failed to initialize detection system.")
        return
    
    print("2. Setting up LLM interface...")
    
    # Choose your LLM interface
    llm_options = {
        '1': ('Ollama', lambda: OllamaInterface("llama3.1")),
        '2': ('LM Studio', lambda: LMStudioInterface("local-model")),
        '3': ('Hugging Face', lambda: HuggingFaceInterface("microsoft/DialoGPT-medium")),
        '4': ('Custom API', lambda: CustomAPIInterface("custom-model", "http://localhost:8080"))
    }
    
    print("Choose LLM interface:")
    for key, (name, _) in llm_options.items():
        print(f"  {key}. {name}")
    
    choice = input("Enter choice (1-4, default 1): ").strip() or '1'
    
    if choice in llm_options:
        name, interface_factory = llm_options[choice]
        llm_interface = interface_factory()
        print(f"Selected: {name}")
    else:
        print("Invalid choice, using Ollama")
        llm_interface = OllamaInterface()
    
    # Test connection
    print("3. Testing LLM connection...")
    if not llm_interface.test_connection():
        print("❌ Cannot connect to LLM. Please check your setup.")
        return
    print("✅ LLM connection successful")
    
    # Initialize tester
    tester = LLMMemorizationTester(llm_interface, detector)
    
    # Get test prompts
    test_prompts = create_test_prompts()
    
    print("4. Running memorization tests...")
    
    # Run different types of tests
    for test_type, prompts in test_prompts.items():
        print(f"\n--- Running {test_type} ---")
        
        # Take only first 3 prompts of each type for demo
        sample_prompts = prompts[:3]
        
        results = tester.test_memorization_with_prompts(
            sample_prompts,
            temperature=0.7,
            max_tokens=200
        )
    
    # Generate report
    print("5. Generating report...")
    report = tester.generate_memorization_report()
    
    print("\n🎯 Testing complete!")

if __name__ == "__main__":
    main()
