#!/usr/bin/env python3
"""
ΨQRH Production Grade GLUE Benchmark
====================================

Benchmark real usando datasets GLUE genuínos com a implementação production-grade.
Métricas reais, sem simulações, usando datasets da biblioteca datasets.

Tarefas implementadas:
- SST-2 (Sentiment Analysis)
- QNLI (Question-answering Natural Language Inference)
- QQP (Quora Question Pairs)
- MNLI (Multi-Genre Natural Language Inference)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import logging
import json
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score
import sys
import os

# Adicionar diretório atual ao path para importar a implementação production-grade
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ΨQRH_PRODUCTION_GRADE import (
    ProductionPsiQrhTransformer,
    ProductionTrainingSystem,
    set_seed
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('glue_benchmark_production.log')
    ]
)

logger = logging.getLogger(__name__)

# =============================================================================
# DATASET LOADER PARA GLUE
# =============================================================================

class RealGLUEDataset(Dataset):
    """
    Dataset real do GLUE usando a biblioteca datasets.
    Carrega dados genuínos das tarefas GLUE.
    """
    
    def __init__(self, task_name: str, split: str = 'validation', max_samples: Optional[int] = None):
        self.task_name = task_name
        self.split = split
        self.max_samples = max_samples
        self.texts, self.labels = self._load_real_data()
        
        logger.info(f"Loaded {len(self.texts)} samples for {task_name} {split}")
    
    def _load_real_data(self) -> Tuple[List[str], List[int]]:
        """Carrega dados reais do GLUE usando datasets library"""
        try:
            from datasets import load_dataset
            
            if self.task_name == 'sst2':
                dataset = load_dataset('glue', 'sst2', split=self.split)
                texts = [item['sentence'] for item in dataset]
                labels = [item['label'] for item in dataset]
                
            elif self.task_name == 'qnli':
                dataset = load_dataset('glue', 'qnli', split=self.split)
                texts = [f"{item['question']} [SEP] {item['sentence']}" for item in dataset]
                labels = [item['label'] for item in dataset]
                
            elif self.task_name == 'qqp':
                dataset = load_dataset('glue', 'qqp', split=self.split)
                texts = [f"{item['question1']} [SEP] {item['question2']}" for item in dataset]
                labels = [item['label'] for item in dataset]
                
            elif self.task_name == 'mnli':
                split_name = 'validation_matched' if self.split == 'validation' else self.split
                dataset = load_dataset('glue', 'mnli', split=split_name)
                texts = [f"{item['premise']} [SEP] {item['hypothesis']}" for item in dataset]
                labels = [item['label'] for item in dataset]
                
            else:
                raise ValueError(f"Tarefa GLUE não suportada: {self.task_name}")
            
            # Limitar número de amostras se especificado
            if self.max_samples and len(texts) > self.max_samples:
                texts = texts[:self.max_samples]
                labels = labels[:self.max_samples]
                
            return texts, labels
            
        except ImportError:
            logger.error("Biblioteca 'datasets' não encontrada. Instale com: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Erro ao carregar dados GLUE: {e}")
            raise
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# =============================================================================
# TOKENIZER SIMPLES PARA GLUE
# =============================================================================

class SimpleGLUETokenizer:
    """
    Tokenizador simples para tarefas GLUE.
    Converte texto para IDs de tokens usando vocabulário básico.
    """
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.vocab = self._build_vocab()
        
    def _build_vocab(self) -> Dict[str, int]:
        """Constrói vocabulário básico"""
        vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<sep>': 2,
        }
        
        # Adicionar caracteres ASCII básicos
        for i, char in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()[]{}"):
            vocab[char] = i + 3
            
        return vocab
    
    def encode(self, text: str) -> List[int]:
        """Codifica texto para IDs de tokens"""
        tokens = []
        for char in text.lower():
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab['<unk>'])
        
        # Truncar ou preencher para max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.vocab['<pad>']] * (self.max_length - len(tokens))
            
        return tokens
    
    def batch_encode(self, texts: List[str]) -> torch.Tensor:
        """Codifica lote de textos"""
        encoded = [self.encode(text) for text in texts]
        return torch.tensor(encoded, dtype=torch.long)

# =============================================================================
# SISTEMA DE BENCHMARK GLUE
# =============================================================================

class ProductionGLUEBenchmark:
    """
    Sistema de benchmark GLUE para a implementação production-grade.
    Avalia o modelo em múltiplas tarefas GLUE com métricas reais.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.tokenizer = SimpleGLUETokenizer()
        self.results = {}
        
        logger.info(f"Initialized GLUE benchmark on device: {self.device}")
    
    def get_num_classes(self, task_name: str) -> int:
        """Retorna número de classes para cada tarefa GLUE"""
        task_classes = {
            'sst2': 2,      # Sentiment (positive/negative)
            'qnli': 2,      # Entailment (entailment/not_entailment)
            'qqp': 2,       # Paraphrase (duplicate/not_duplicate)
            'mnli': 3,      # Entailment (entailment/neutral/contradiction)
        }
        return task_classes.get(task_name, 2)
    
    def create_model(self, task_name: str) -> ProductionPsiQrhTransformer:
        """Cria modelo para tarefa específica"""
        num_classes = self.get_num_classes(task_name)
        
        model = ProductionPsiQrhTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=256,
            n_layers=6,
            n_heads=8,
            num_classes=num_classes,
            max_seq_len=128
        ).to(self.device)
        
        logger.info(f"Created model for {task_name} with {num_classes} classes")
        return model
    
    def evaluate_task(self, task_name: str, num_epochs: int = 3) -> Dict[str, float]:
        """
        Avalia modelo em tarefa GLUE específica.
        Retorna métricas reais de accuracy e F1.
        """
        logger.info(f"Evaluating on {task_name}...")
        
        # Carregar dados
        try:
            train_dataset = RealGLUEDataset(task_name, 'train', max_samples=1000)
            val_dataset = RealGLUEDataset(task_name, 'validation', max_samples=200)
        except Exception as e:
            logger.error(f"Failed to load data for {task_name}: {e}")
            return {'accuracy': 0.0, 'f1': 0.0, 'error': str(e)}
        
        # Criar data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Criar e treinar modelo
        model = self.create_model(task_name)
        
        # Criar data loaders customizados que tokenizam os dados
        def create_tokenized_loader(dataset, batch_size=8, shuffle=False):
            texts = [dataset[i][0] for i in range(len(dataset))]
            labels = [dataset[i][1] for i in range(len(dataset))]
            
            # Tokenizar todos os dados de uma vez
            input_ids = self.tokenizer.batch_encode(texts)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            # Criar dataset tokenizado
            tokenized_dataset = torch.utils.data.TensorDataset(input_ids, labels_tensor)
            return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle)
        
        # Criar data loaders tokenizados
        train_loader_tokenized = create_tokenized_loader(train_dataset, batch_size=8, shuffle=True)
        val_loader_tokenized = create_tokenized_loader(val_dataset, batch_size=16, shuffle=False)
        
        # Sistema de treinamento
        training_system = ProductionTrainingSystem(
            model=model,
            train_loader=train_loader_tokenized,
            val_loader=val_loader_tokenized,
            learning_rate=5e-5,
            weight_decay=0.01
        )
        
        # Treinar modelo
        logger.info(f"Training on {task_name} for {num_epochs} epochs...")
        training_system.train(num_epochs=num_epochs)
        
        # Avaliar no conjunto de validação
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for input_ids, labels in val_loader_tokenized:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                logits = model(input_ids)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calcular métricas
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        metrics = {
            'accuracy': float(accuracy),
            'f1': float(f1),
            'num_samples': len(all_labels)
        }
        
        logger.info(f"{task_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def run_full_benchmark(self, tasks: List[str] = None, num_epochs: int = 3) -> Dict[str, Dict]:
        """
        Executa benchmark completo em todas as tarefas especificadas.
        """
        if tasks is None:
            tasks = ['sst2', 'qnli', 'qqp', 'mnli']
        
        logger.info(f"Starting full GLUE benchmark on tasks: {tasks}")
        
        results = {}
        total_accuracy = 0.0
        total_f1 = 0.0
        valid_tasks = 0
        
        for task in tasks:
            try:
                task_results = self.evaluate_task(task, num_epochs=num_epochs)
                results[task] = task_results
                
                if 'error' not in task_results:
                    total_accuracy += task_results['accuracy']
                    total_f1 += task_results['f1']
                    valid_tasks += 1
                    
            except Exception as e:
                logger.error(f"Failed to evaluate {task}: {e}")
                results[task] = {'error': str(e)}
        
        # Calcular média GLUE
        if valid_tasks > 0:
            avg_accuracy = total_accuracy / valid_tasks
            avg_f1 = total_f1 / valid_tasks
            glue_score = (avg_accuracy + avg_f1) / 2
        else:
            avg_accuracy = avg_f1 = glue_score = 0.0
        
        # Resultados sumarizados
        summary = {
            'glue_score': float(glue_score),
            'average_accuracy': float(avg_accuracy),
            'average_f1': float(avg_f1),
            'valid_tasks': valid_tasks,
            'total_tasks': len(tasks)
        }
        
        self.results = {
            'summary': summary,
            'task_results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': {
                'name': 'ProductionPsiQrhTransformer',
                'd_model': 256,
                'n_layers': 6,
                'n_heads': 8
            }
        }
        
        self._print_results()
        self._save_results()
        
        return self.results
    
    def _print_results(self):
        """Imprime resultados do benchmark"""
        print("\n" + "="*80)
        print("ΨQRH PRODUCTION GRADE GLUE BENCHMARK RESULTS")
        print("="*80)
        
        summary = self.results['summary']
        task_results = self.results['task_results']
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   GLUE Score:     {summary['glue_score']:.4f}")
        print(f"   Avg Accuracy:   {summary['average_accuracy']:.4f}")
        print(f"   Avg F1:         {summary['average_f1']:.4f}")
        print(f"   Tasks Completed: {summary['valid_tasks']}/{summary['total_tasks']}")
        
        print(f"\n🔬 TASK-WISE RESULTS:")
        for task, results in task_results.items():
            if 'error' in results:
                print(f"   {task.upper():8} - ERROR: {results['error']}")
            else:
                print(f"   {task.upper():8} - Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")
        
        print("\n⚙️  MODEL CONFIGURATION:")
        model_info = self.results['model_info']
        print(f"   Model: {model_info['name']}")
        print(f"   d_model: {model_info['d_model']}, Layers: {model_info['n_layers']}, Heads: {model_info['n_heads']}")
        
        print("="*80)
    
    def _save_results(self):
        """Salva resultados em arquivo JSON"""
        filename = f"glue_benchmark_production_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """Função principal para executar benchmark GLUE"""
    print("🚀 ΨQRH Production Grade GLUE Benchmark")
    print("========================================")
    print("📊 Real GLUE datasets, real metrics, no simulations")
    print("🔬 Production-grade implementation evaluation")
    print("="*60)
    
    # Configurar device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configurar seed para reprodutibilidade
    set_seed(42)
    
    # Criar e executar benchmark
    benchmark = ProductionGLUEBenchmark(device=device)
    
    # Tarefas GLUE a avaliar
    tasks = ['sst2', 'qnli', 'qqp', 'mnli']
    
    try:
        results = benchmark.run_full_benchmark(tasks=tasks, num_epochs=3)
        
        print("\n🎉 BENCHMARK COMPLETED SUCCESSFULLY!")
        print(f"📈 Final GLUE Score: {results['summary']['glue_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        print(f"\n❌ BENCHMARK FAILED: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())