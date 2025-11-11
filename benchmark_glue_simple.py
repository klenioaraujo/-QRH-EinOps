#!/usr/bin/env python3
"""
ΨQRH Simple GLUE Benchmark - Version Simplificada
==================================================

Benchmark simplificado que usa apenas avaliação direta sem treinamento
para validar a implementação production-grade com métricas reais.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
import json
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, f1_score
import sys
import os

# Adicionar diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ΨQRH_PRODUCTION_GRADE import (
    ProductionPsiQrhTransformer,
    set_seed
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleGLUEEvaluator:
    """Avaliador simples para GLUE usando apenas inferência"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.tokenizer = SimpleTokenizer()
        
    def evaluate_task(self, task_name: str, num_samples: int = 100) -> Dict[str, float]:
        """Avalia modelo em tarefa GLUE com dados sintéticos simples"""
        logger.info(f"Evaluating {task_name} with synthetic data...")
        
        # Criar modelo
        num_classes = self.get_num_classes(task_name)
        model = ProductionPsiQrhTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=256,
            n_layers=6,
            n_heads=8,
            num_classes=num_classes,
            max_seq_len=128
        ).to(self.device)
        
        # Gerar dados sintéticos simples
        texts, labels = self.generate_synthetic_data(task_name, num_samples)
        
        # Tokenizar
        input_ids = self.tokenizer.batch_encode(texts).to(self.device)
        true_labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        # Inferência com processamento individual para evitar problemas de batch
        model.eval()
        all_predictions = []
        
        with torch.no_grad():
            # Processar cada amostra individualmente
            for i in range(len(input_ids)):
                single_input = input_ids[i:i+1]  # [1, seq_len]
                single_logits = model(single_input)  # [1, num_classes]
                single_pred = torch.argmax(single_logits, dim=1)  # [1]
                all_predictions.append(single_pred.item())
        
        predictions = torch.tensor(all_predictions)
        
        # Calcular métricas
        pred_np = predictions.cpu().numpy()
        true_np = true_labels.cpu().numpy()
        
        accuracy = accuracy_score(true_np, pred_np)
        f1 = f1_score(true_np, pred_np, average='weighted')
        
        return {
            'accuracy': float(accuracy),
            'f1': float(f1),
            'num_samples': num_samples,
            'task': task_name
        }
    
    def get_num_classes(self, task_name: str) -> int:
        task_classes = {
            'sst2': 2, 'qnli': 2, 'qqp': 2, 'mnli': 3
        }
        return task_classes.get(task_name, 2)
    
    def generate_synthetic_data(self, task_name: str, num_samples: int) -> Tuple[List[str], List[int]]:
        """Gera dados sintéticos simples para teste"""
        if task_name == 'sst2':
            # Sentiment analysis
            positive_texts = ["This is great", "I love this", "Amazing product", "Excellent quality"]
            negative_texts = ["This is bad", "I hate this", "Terrible product", "Poor quality"]
            texts = (positive_texts * (num_samples // 2)) + (negative_texts * (num_samples // 2))
            labels = [1] * (num_samples // 2) + [0] * (num_samples // 2)
            
        elif task_name == 'qnli':
            # Question answering
            texts = [
                "What is AI? [SEP] Artificial intelligence is technology",
                "How does it work? [SEP] It works using algorithms",
                "What is ML? [SEP] Machine learning is a subset of AI"
            ] * (num_samples // 3)
            labels = [0, 1, 0] * (num_samples // 3)
            
        elif task_name == 'qqp':
            # Question pairs
            texts = [
                "What is AI? [SEP] What is artificial intelligence?",
                "How to code? [SEP] How to program?",
                "What is ML? [SEP] What is deep learning?"
            ] * (num_samples // 3)
            labels = [1, 1, 0] * (num_samples // 3)
            
        elif task_name == 'mnli':
            # Natural language inference
            texts = [
                "The cat sat [SEP] The cat was sitting",
                "It was raining [SEP] The sun was shining",
                "He was happy [SEP] He felt joyful"
            ] * (num_samples // 3)
            labels = [0, 2, 0] * (num_samples // 3)
            
        else:
            texts = ["Test text"] * num_samples
            labels = [0] * num_samples
            
        return texts[:num_samples], labels[:num_samples]

class SimpleTokenizer:
    """Tokenizador simples"""
    
    def __init__(self, vocab_size=1000, max_length=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.vocab = self._build_vocab()
    
    def _build_vocab(self):
        vocab = {'<pad>': 0, '<unk>': 1}
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()[]{}"
        for i, char in enumerate(chars):
            vocab[char] = i + 2
        return vocab
    
    def encode(self, text: str) -> List[int]:
        tokens = []
        for char in text.lower():
            tokens.append(self.vocab.get(char, self.vocab['<unk>']))
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens += [self.vocab['<pad>']] * (self.max_length - len(tokens))
        return tokens
    
    def batch_encode(self, texts: List[str]) -> torch.Tensor:
        encoded = [self.encode(text) for text in texts]
        return torch.tensor(encoded, dtype=torch.long)

def run_simple_benchmark():
    """Executa benchmark simplificado"""
    print("🚀 ΨQRH Simple GLUE Benchmark")
    print("==============================")
    print("📊 Synthetic data, real metrics, production-grade model")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    set_seed(42)
    
    evaluator = SimpleGLUEEvaluator(device=device)
    tasks = ['sst2', 'qnli', 'qqp', 'mnli']
    
    results = {}
    total_accuracy = 0.0
    valid_tasks = 0
    
    for task in tasks:
        try:
            task_results = evaluator.evaluate_task(task, num_samples=100)
            results[task] = task_results
            total_accuracy += task_results['accuracy']
            valid_tasks += 1
            print(f"✅ {task.upper():8} - Accuracy: {task_results['accuracy']:.4f}")
            
        except Exception as e:
            print(f"❌ {task.upper():8} - ERROR: {e}")
            results[task] = {'error': str(e)}
    
    # Calcular média
    if valid_tasks > 0:
        avg_accuracy = total_accuracy / valid_tasks
    else:
        avg_accuracy = 0.0
    
    # Resultados finais
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Tasks Completed:  {valid_tasks}/{len(tasks)}")
    print("="*60)
    
    # Salvar resultados
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'average_accuracy': avg_accuracy,
        'tasks_evaluated': valid_tasks,
        'task_results': results,
        'model': 'ProductionPsiQrhTransformer'
    }
    
    filename = f"simple_glue_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {filename}")
    return output

if __name__ == "__main__":
    results = run_simple_benchmark()
    
    if results['tasks_evaluated'] > 0:
        print("\n🎉 Benchmark completed successfully!")
        print("Production-grade model is operational with real metrics.")
    else:
        print("\n❌ Benchmark failed - check model implementation.")