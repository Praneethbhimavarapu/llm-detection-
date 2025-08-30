#!/usr/bin/env python3
"""
Main LLM Memorization Detection System
Multi-modal detection supporting text, images, and documents
"""

import os
import json
import hashlib
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from .content_extractor import ContentExtractor
    from .similarity_algorithms import TextProcessor, SimilarityCalculator
    from .utils import MemorizationResult, safe_import
except ImportError:
    # Handle direct script execution
    from content_extractor import ContentExtractor
    from similarity_algorithms import TextProcessor, SimilarityCalculator
    from utils import MemorizationResult, safe_import


class MemorizationDetector:
    """Core memorization detection engine"""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.training_data = {
            'text': [],
            'image': [],
            'document': []
        }
        self.content_hashes = {
            'text': set(),
            'image': set(),
            'document': set()
        }
        self.processor = TextProcessor()
        self.similarity_calc = SimilarityCalculator()
    
    def add_training_content(self, content_data: Dict[str, Any], source: str = None):
        """Add content to training dataset"""
        content = content_data.get('content', '')
        content_type = content_data.get('content_type', 'text')
        
        if not content:
            return
        
        content_hash = hashlib.sha256(str(content).encode('utf-8')).hexdigest()
        
        if content_hash in self.content_hashes.get(content_type, set()):
            return
        
        training_item = {
            'content': content,
            'hash': content_hash,
            'source': source,
            'metadata': content_data.get('metadata', {})
        }
        
        # Add type-specific processing
        if content_type == 'text':
            training_item['cleaned'] = self.processor.clean_text(content)
            training_item['words'] = set(training_item['cleaned'].split())
            training_item['ngrams'] = set(self.processor.extract_ngrams(training_item['cleaned'], 3))
        elif content_type == 'image':
            training_item['visual_hash'] = content_data.get('visual_hash', content_hash)
            training_item['file_hash'] = content_data.get('file_hash', content_hash)
        
        self.training_data[content_type].append(training_item)
        self.content_hashes[content_type].add(content_hash)
        
        logger.debug(f"Added {content_type} content from: {source}")
    
    def detect_memorization(self, content_data: Dict[str, Any]) -> MemorizationResult:
        """Detect memorization in content"""
        content = content_data.get('content', '')
        content_type = content_data.get('content_type', 'text')
        
        if not content:
            return MemorizationResult("", 0.0, False, 0.0, detection_method="empty_content")
        
        content_hash = hashlib.sha256(str(content).encode('utf-8')).hexdigest()
        
        # Check exact matches first
        if content_hash in self.content_hashes.get(content_type, set()):
            matched_item = next(
                item for item in self.training_data[content_type] 
                if item['hash'] == content_hash
            )
            return MemorizationResult(
                content_hash=content_hash,
                similarity_score=1.0,
                is_memorized=True,
                confidence=1.0,
                matched_content=str(matched_item['content'])[:200] + "...",
                source_file=matched_item['source'],
                detection_method="exact_hash_match"
            )
        
        # Type-specific similarity detection
        if content_type == 'text':
            return self._detect_text_similarity(content, content_hash)
        elif content_type == 'image':
            return self._detect_image_similarity(content_data, content_hash)
        else:
            return self._detect_document_similarity(content, content_hash, content_type)
    
    def _detect_text_similarity(self, content: str, content_hash: str) -> MemorizationResult:
        """Detect text similarity using multiple algorithms"""
        cleaned_content = self.processor.clean_text(content)
        content_words = set(cleaned_content.split())
        content_ngrams = set(self.processor.extract_ngrams(cleaned_content, 3))
        
        best_similarity = 0.0
        best_match = None
        best_method = "no_match"
        
        for item in self.training_data['text']:
            # Multiple similarity measures
            similarities = {}
            
            # Jaccard similarity on words
            similarities['jaccard_words'] = self.similarity_calc.jaccard_similarity(
                content_words, item['words']
            )
            
            # N-gram similarity
            similarities['ngram'] = self.similarity_calc.jaccard_similarity(
                content_ngrams, item.get('ngrams', set())
            )
            
            # Cosine similarity (if available)
            try:
                similarities['cosine'] = self.similarity_calc.cosine_similarity(
                    cleaned_content, item['cleaned']
                )
            except:
                similarities['cosine'] = 0.0
            
            # Combined score with weights
            combined_similarity = (
                similarities['jaccard_words'] * 0.4 +
                similarities['ngram'] * 0.4 +
                similarities['cosine'] * 0.2
            )
            
            if combined_similarity > best_similarity:
                best_similarity = combined_similarity
                best_match = item
                best_method = "combined_similarity"
        
        is_memorized = best_similarity > self.threshold
        
        return MemorizationResult(
            content_hash=content_hash,
            similarity_score=best_similarity,
            is_memorized=is_memorized,
            confidence=best_similarity if is_memorized else (1.0 - best_similarity),
            matched_content=best_match['content'][:200] + "..." if best_match else None,
            source_file=best_match['source'] if best_match else None,
            detection_method=best_method
        )
    
    def _detect_image_similarity(self, content_data: Dict[str, Any], content_hash: str) -> MemorizationResult:
        """Detect image similarity"""
        visual_hash = content_data.get('visual_hash', content_hash)
        file_hash = content_data.get('file_hash', content_hash)
        
        # Check visual similarity
        for item in self.training_data['image']:
            if item.get('visual_hash') == visual_hash:
                return MemorizationResult(
                    content_hash=content_hash,
                    similarity_score=1.0,
                    is_memorized=True,
                    confidence=1.0,
                    matched_content=f"Visual match: {item['source']}",
                    source_file=item['source'],
                    detection_method="visual_similarity"
                )
            
            if item.get('file_hash') == file_hash:
                return MemorizationResult(
                    content_hash=content_hash,
                    similarity_score=1.0,
                    is_memorized=True,
                    confidence=1.0,
                    matched_content=f"Exact file match: {item['source']}",
                    source_file=item['source'],
                    detection_method="file_duplicate"
                )
        
        return MemorizationResult(
            content_hash=content_hash,
            similarity_score=0.0,
            is_memorized=False,
            confidence=0.8,
            detection_method="no_image_match"
        )
    
    def _detect_document_similarity(self, content: str, content_hash: str, content_type: str) -> MemorizationResult:
        """Detect document similarity"""
        if content.strip():
            return self._detect_text_similarity(content, content_hash)
        
        return MemorizationResult(
            content_hash=content_hash,
            similarity_score=0.0,
            is_memorized=False,
            confidence=0.5,
            detection_method="empty_document"
        )


class CompleteLLMMemorizationSystem:
    """Complete system with all features"""
    
    def __init__(self, threshold: float = 0.85, config_path: str = None):
        self.threshold = threshold
        self.config = self._load_config(config_path) if config_path else {}
        
        self.extractor = ContentExtractor()
        self.detector = MemorizationDetector(threshold)
        self.system_ready = False
        self.stats = {'processed': 0, 'memorized': 0, 'errors': 0}
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}
    
    def load_training_directory(self, directory: str):
        """Load all files from training directory"""
        data_path = Path(directory)
        
        if not data_path.exists():
            logger.error(f"Directory {directory} not found")
            return
        
        file_counts = defaultdict(int)
        
        for file_path in data_path.rglob('*'):
            if file_path.is_file():
                try:
                    content_data = self.extractor.extract_content(str(file_path))
                    content_type = content_data.get('content_type', 'unknown')
                    
                    if content_type != 'unsupported':
                        self.detector.add_training_content(content_data, str(file_path))
                        file_counts[content_type] += 1
                    else:
                        file_counts['unsupported'] += 1
                
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    file_counts['errors'] += 1
        
        self.system_ready = True
        
        logger.info(f"Training Data Loaded:")
        for ctype, count in file_counts.items():
            logger.info(f"  {ctype.title()}: {count} files")
    
    def analyze_content(self, input_item: str) -> MemorizationResult:
        """Analyze text or file for memorization"""
        if Path(input_item).exists():
            # File input
            content_data = self.extractor.extract_content(input_item)
            result = self.detector.detect_memorization(content_data)
        else:
            # Text input
            content_data = {'content': input_item, 'content_type': 'text', 'metadata': {}}
            result = self.detector.detect_memorization(content_data)
        
        self.stats['processed'] += 1
        if result.is_memorized:
            self.stats['memorized'] += 1
        
        return result
    
    def batch_analyze(self, items: List[str]) -> List[MemorizationResult]:
        """Analyze multiple items"""
        results = []
        for item in items:
            try:
                result = self.analyze_content(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {item}: {e}")
                self.stats['errors'] += 1
                # Add error result
                results.append(MemorizationResult(
                    content_hash="",
                    similarity_score=0.0,
                    is_memorized=False,
                    confidence=0.0,
                    detection_method="processing_error",
                    matched_content=str(e)
                ))
        
        return results
    
    def export_results(self, results: List[MemorizationResult], output_file: str):
        """Export results to JSON file"""
        data = {
            'system_stats': self.stats,
            'results': [r.to_dict() for r in results],
            'configuration': {
                'threshold': self.detector.threshold,
                'supported_types': list(self.extractor.supported_extensions.keys())
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results exported to {output_file}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        training_stats = {}
        for content_type, items in self.detector.training_data.items():
            training_stats[content_type] = len(items)
        
        return {
            'processing_stats': self.stats,
            'training_data_stats': training_stats,
            'configuration': {
                'threshold': self.detector.threshold,
                'system_ready': self.system_ready
            }
        }


def create_sample_data():
    """Create sample training data for testing"""
    data_dir = Path("./data/sample_training_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample texts
    texts = [
        "The transformer architecture revolutionized natural language processing in 2017",
        "Convolutional neural networks excel at computer vision tasks",
        "BERT introduced bidirectional training for language models",
        "GPT models use autoregressive generation for text completion",
        "Attention mechanisms allow models to focus on relevant information"
    ]
    
    for i, text in enumerate(texts):
        with open(data_dir / f"sample_text_{i+1}.txt", 'w') as f:
            f.write(text)
    
    # Sample JSON data
    ai_data = {
        "models": [
            {"name": "GPT-3", "parameters": "175B", "company": "OpenAI"},
            {"name": "BERT", "parameters": "340M", "company": "Google"},
            {"name": "T5", "parameters": "11B", "company": "Google"}
        ],
        "techniques": ["attention", "transformer", "fine-tuning", "pre-training"]
    }
    
    with open(data_dir / "ai_models.json", 'w') as f:
        json.dump(ai_data, f, indent=2)
    
    logger.info(f"Created sample training data in {data_dir}")
    return str(data_dir)


def main():
    """Main demonstration"""
    print("LLM Memorization Detection System")
    print("=" * 40)
    
    # Create sample data
    training_dir = create_sample_data()
    
    # Initialize system
    system = CompleteLLMMemorizationSystem(threshold=0.75)
    system.load_training_directory(training_dir)
    
    # Test cases
    test_cases = [
        "The transformer architecture revolutionized",  # Should match
        "BERT and GPT models are powerful",            # Should partially match
        "Random new content that is unique",           # Should not match
        "Attention mechanisms in neural networks",     # Should partially match
    ]
    
    print("\nTest Results:")
    print("-" * 40)
    results = []
    
    for test in test_cases:
        result = system.analyze_content(test)
        results.append(result)
        
        status = "MEMORIZED" if result.is_memorized else "ORIGINAL"
        print(f"\nText: {test}")
        print(f"Status: {status}")
        print(f"Score: {result.similarity_score:.4f}")
        print(f"Method: {result.detection_method}")
        
        if result.matched_content:
            print(f"Match: {result.matched_content[:60]}...")
    
    # Export results
    system.export_results(results, "sample_results.json")
    
    # Display statistics
    stats = system.get_system_stats()
    print(f"\nSystem Statistics:")
    print(f"Total Processed: {stats['processing_
