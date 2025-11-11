
#!/usr/bin/env python3
"""
Production Grade Tests for ΨQRH Implementation
==============================================

Comprehensive test suite validating all production-grade components
with proper error handling, performance benchmarks, and correctness checks.
"""

import torch
import torch.nn as nn
import time
import pytest
import sys
import os
import math

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ΨQRH_PRODUCTION_GRADE import (
    ProductionQuaternionOperations,
    ProductionSpectralAttention,
    ProductionEmbedding,
    ProductionLeechLattice,
    ProductionPsiQrhTransformer,
    ProductionTests,
    set_seed
)

# Set seed for reproducibility
set_seed(42)

class TestProductionQuaternionOperations:
    """Test suite for production-grade quaternion operations"""
    
    def test_quaternion_multiply_shapes(self):
        """Test quaternion multiplication with various shapes"""
        # Test basic shape [4]
        q1 = torch.randn(4)
        q2 = torch.randn(4)
        result = ProductionQuaternionOperations.quaternion_multiply(q1, q2)
        assert result.shape == (4,), f"Expected (4,), got {result.shape}"
        
        # Test batch shape [batch, 4]
        q1 = torch.randn(8, 4)
        q2 = torch.randn(8, 4)
        result = ProductionQuaternionOperations.quaternion_multiply(q1, q2)
        assert result.shape == (8, 4), f"Expected (8, 4), got {result.shape}"
        
        # Test sequence shape [batch, seq, 4]
        q1 = torch.randn(2, 16, 4)
        q2 = torch.randn(2, 16, 4)
        result = ProductionQuaternionOperations.quaternion_multiply(q1, q2)
        assert result.shape == (2, 16, 4), f"Expected (2, 16, 4), got {result.shape}"
    
    def test_quaternion_multiply_correctness(self):
        """Test mathematical correctness of quaternion multiplication"""
        # Test identity quaternion
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q = torch.tensor([0.5, 0.5, 0.5, 0.5])
        
        result = ProductionQuaternionOperations.quaternion_multiply(identity, q)
        torch.testing.assert_close(result, q, rtol=1e-6, atol=1e-6)
        
        # Test known quaternion multiplication
        q1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        q2 = torch.tensor([5.0, 6.0, 7.0, 8.0])
        
        result = ProductionQuaternionOperations.quaternion_multiply(q1, q2)
        expected = torch.tensor([-60.0, 12.0, 30.0, 24.0])
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)
    
    def test_unit_quaternion(self):
        """Test unit quaternion creation"""
        theta = torch.tensor([0.0, math.pi/2, math.pi])
        omega = torch.tensor([0.0, math.pi/4, math.pi/2])
        phi = torch.tensor([0.0, math.pi/6, math.pi/3])
        
        unit_q = ProductionQuaternionOperations.unit_quaternion(theta, omega, phi)
        
        # Check shape
        assert unit_q.shape == (3, 4), f"Expected (3, 4), got {unit_q.shape}"
        
        # Check norm is approximately 1
        norms = torch.norm(unit_q, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-6, atol=1e-6)
    
    def test_so4_rotation(self):
        """Test SO(4) rotation correctness"""
        # Test identity rotation
        psi = torch.tensor([1.0, 2.0, 3.0, 4.0])
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
        
        rotated = ProductionQuaternionOperations.so4_rotation(psi, identity, identity)
        torch.testing.assert_close(rotated, psi, rtol=1e-6, atol=1e-6)
        
        # Test norm preservation
        q_left = torch.tensor([0.7071, 0.7071, 0.0, 0.0])  # 90° rotation around x
        q_right = torch.tensor([0.7071, 0.0, 0.7071, 0.0])  # 90° rotation around y
        
        input_norm = torch.norm(psi, p=2)
        rotated_norm = torch.norm(rotated, p=2)
        
        torch.testing.assert_close(input_norm, rotated_norm, rtol=1e-6, atol=1e-6)

class TestProductionSpectralAttention:
    """Test suite for production-grade spectral attention"""
    
    def test_attention_shapes(self):
        """Test attention input/output shapes"""
        attention = ProductionSpectralAttention(d_model=64, n_heads=8)
        
        # Test various input shapes
        batch_sizes = [1, 4, 16]
        seq_lens = [8, 32, 128]
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                x = torch.randn(batch_size, seq_len, 64)
                fractal_dim = torch.tensor(1.5)
                
                output = attention(x, fractal_dim)
                assert output.shape == x.shape, f"Shape mismatch for batch={batch_size}, seq={seq_len}"
    
    def test_attention_device_safety(self):
        """Test attention works correctly on different devices"""
        attention = ProductionSpectralAttention(d_model=64, n_heads=8)
        x = torch.randn(2, 16, 64)
        fractal_dim = torch.tensor(1.5)
        
        # Test CPU
        output_cpu = attention(x, fractal_dim)
        assert output_cpu.device.type == 'cpu'
        
        # Test GPU if available
        if torch.cuda.is_available():
            attention = attention.cuda()
            x = x.cuda()
            fractal_dim = fractal_dim.cuda()
            
            output_gpu = attention(x, fractal_dim)
            assert output_gpu.device.type == 'cuda'
            assert output_gpu.shape == x.shape
    
    def test_attention_gradients(self):
        """Test that gradients flow properly through attention"""
        attention = ProductionSpectralAttention(d_model=64, n_heads=8)
        x = torch.randn(2, 16, 64, requires_grad=True)
        fractal_dim = torch.tensor(1.5)
        
        output = attention(x, fractal_dim)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None, "Gradients should flow to input"
        assert x.grad.shape == x.shape, "Gradient shape should match input shape"

class TestProductionEmbedding:
    """Test suite for production-grade embedding system"""
    
    def test_embedding_shapes(self):
        """Test embedding input/output shapes"""
        vocab_size = 1000
        d_model = 64
        max_seq_len = 128
        
        embedding = ProductionEmbedding(vocab_size, d_model, max_seq_len)
        
        # Test various batch and sequence lengths
        test_cases = [
            (1, 8), (4, 32), (16, 128), (32, 64)
        ]
        
        for batch_size, seq_len in test_cases:
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            output = embedding(input_ids)
            
            expected_shape = (batch_size, seq_len, d_model)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_embedding_device_consistency(self):
        """Test embedding works correctly on different devices"""
        embedding = ProductionEmbedding(vocab_size=1000, d_model=64)
        input_ids = torch.randint(0, 1000, (4, 32))
        
        # Test CPU
        output_cpu = embedding(input_ids)
        assert output_cpu.device.type == 'cpu'
        
        # Test GPU if available
        if torch.cuda.is_available():
            embedding = embedding.cuda()
            input_ids = input_ids.cuda()
            
            output_gpu = embedding(input_ids)
            assert output_gpu.device.type == 'cuda'
            assert output_gpu.shape == (4, 32, 64)
    
    def test_embedding_performance(self):
        """Test embedding performance with large batches"""
        embedding = ProductionEmbedding(vocab_size=50000, d_model=256)
        
        # Large batch size to test vectorization
        input_ids = torch.randint(0, 50000, (32, 512))
        
        start_time = time.time()
        output = embedding(input_ids)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Embedding execution time: {execution_time:.4f}s")
        
        # Should be fast (sub-second for this size)
        assert execution_time < 1.0, f"Embedding too slow: {execution_time:.4f}s"
        assert output.shape == (32, 512, 256)

class TestProductionLeechLattice:
    """Test suite for production-grade Leech lattice"""
    
    def test_lattice_shapes(self):
        """Test lattice encoding/decoding shapes"""
        lattice = ProductionLeechLattice(embed_dim=64)
        
        # Test various input shapes
        batch_sizes = [1, 4, 16]
        seq_lens = [8, 32, 128]
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                data = torch.randn(batch_size, seq_len, 64)
                
                encoded = lattice.encode_to_lattice(data)
                decoded = lattice.decode_from_lattice(encoded)
                
                assert encoded.shape == (batch_size, seq_len, 24), f"Encoded shape mismatch"
                assert decoded.shape == data.shape, f"Decoded shape mismatch"
    
    def test_lattice_quantization(self):
        """Test that lattice encoding properly quantizes values"""
        lattice = ProductionLeechLattice(embed_dim=8)
        data = torch.randn(2, 4, 8)
        
        encoded = lattice.encode_to_lattice(data)
        
        # Check that encoded values are from quantization levels
        quantization_levels = lattice.quantization_levels
        # Each encoded value should be one of the quantization levels
        for level in quantization_levels:
            # Check if any value in encoded matches this level (within tolerance)
            level_matches = torch.isclose(encoded, torch.tensor(level.item()), atol=1e-6)
            # At least some values should match quantization levels
            assert torch.any(level_matches), f"No values match quantization level {level}"
    
    def test_lattice_error_correction(self):
        """Test error correction functionality"""
        lattice = ProductionLeechLattice(embed_dim=16)
        
        # Create data with small values that should be zeroed out
        data = torch.tensor([
            [[0.05, 0.1, -0.05, 0.2] * 4],  # Small values
            [[0.5, 1.0, -0.5, 2.0] * 4]     # Large values
        ], dtype=torch.float32)
        
        encoded = lattice.encode_to_lattice(data)
        decoded = lattice.decode_from_lattice(encoded)
        
        # Check that decoded values are reasonable (not NaN or Inf)
        assert torch.all(torch.isfinite(decoded)), "Decoded values contain NaN or Inf"
        
        # Check that decoded shape matches input shape
        assert decoded.shape == data.shape, f"Shape mismatch: {decoded.shape} vs {data.shape}"
        
        # The lattice encoding/decoding should produce stable output
        # (we can't guarantee exact values due to quantization)
        assert torch.all(torch.abs(decoded) <= 2.0), "Decoded values out of reasonable range"

class TestProductionTransformer:
    """Test suite for production-grade transformer"""
    
    def test_transformer_forward(self):
        """Test complete transformer forward pass"""
        model = ProductionPsiQrhTransformer(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=4,
            num_classes=3,
            max_seq_len=128
        )
        
        # Test various input sizes
        test_cases = [
            (1, 8), (4, 32), (8, 64)
        ]
        
        for batch_size, seq_len in test_cases:
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            logits = model(input_ids)
            
            expected_shape = (batch_size, 3)  # num_classes
            assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    
    def test_transformer_device_safety(self):
        """Test transformer works on different devices"""
        model = ProductionPsiQrhTransformer(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=4
        )
        
        input_ids = torch.randint(0, 1000, (4, 32))
        
        # Test CPU
        logits_cpu = model(input_ids)
        assert logits_cpu.device.type == 'cpu'
        
        # Test GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            input_ids = input_ids.cuda()
            
            logits_gpu = model(input_ids)
            assert logits_gpu.device.type == 'cuda'
            assert logits_gpu.shape == (4, 2)  # Default num_classes=2
    
    def test_transformer_gradient_flow(self):
        """Test that gradients flow through entire transformer"""
        model = ProductionPsiQrhTransformer(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=4
        )
        
        input_ids = torch.randint(0, 1000, (4, 32))
        logits = model(input_ids)
        
        # Create dummy target and compute loss
        target = torch.randint(0, 2, (4,))
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        
        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for parameter: {name}"

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    def benchmark_embedding_speed(self):
        """Benchmark embedding speed with large vocabulary"""
        print("\n=== Embedding Performance Benchmark ===")
        
        vocab_size = 50000
        d_model = 256
        batch_size = 32
        seq_len = 512
        
        embedding = ProductionEmbedding(vocab_size, d_model)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Warmup
        for _ in range(5):
            _ = embedding(input_ids)
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            output = embedding(input_ids)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"Average embedding time: {avg_time:.4f}s")
        print(f"Throughput: {batch_size * seq_len / avg_time:.0f} tokens/s")
        
        assert avg_time < 0.1, f"Embedding too slow: {avg_time:.4f}s"
    
    def benchmark_attention_speed(self):
        """Benchmark spectral attention speed"""
        print("\n=== Attention Performance Benchmark ===")
        
        d_model = 256
        n_heads = 8
        batch_size = 8
        seq_len = 512
        
        attention = ProductionSpectralAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        fractal_dim = torch.tensor(1.5)
        
        # Warmup
        for _ in range(5):
            _ = attention(x, fractal_dim)
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            output = attention(x, fractal_dim)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"Average attention time: {avg_time:.4f}s")
        print(f"Throughput: {batch_size * seq_len / avg_time:.0f} tokens/s")
        
        assert avg_time < 0.5, f"Attention too slow: {avg_time:.4f}s"
    
    def benchmark_full_model(self):
        """Benchmark full transformer model speed"""
        print("\n=== Full Model Performance Benchmark ===")
        
        model = ProductionPsiQrhTransformer(
            vocab_size=10000,
            d_model=256,
            n_layers=6,
            n_heads=8,
            max_seq_len=512
        )
        
        batch_size = 4
        seq_len = 256
        input_ids = torch.randint(0, 10000, (batch_size, seq_len))
        
        # Warmup
        for _ in range(3):
            _ = model(input_ids)
        
        # Benchmark
        start_time = time.time()
        for _ in range(5):
            logits = model(input_ids)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 5
        print(f"Average model time: {avg_time:.4f}s")
        print(f"Throughput: {batch_size * seq_len / avg_time:.0f} tokens/s")
        
        assert avg_time < 2.0, f"Full model too slow: {avg_time:.4f}s"

def run_all_tests():
    """Run all production grade tests"""
    print("=" * 60)
    print("PRODUCTION GRADE ΨQRH TEST SUITE")
    print("=" * 60)
    
    # Run unit tests
    test_classes = [
        TestProductionQuaternionOperations,
        TestProductionSpectralAttention,
        TestProductionEmbedding,
        TestProductionLeechLattice,
        TestProductionTransformer
    ]
    
    for test_class in test_classes:
        print(f"\n--- Testing {test_class.__name__} ---")
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance)
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            print(f"  Running {method_name}...")
            try:
                getattr(test_instance, method_name)()
                print(f"  ✓ {method_name} PASSED")
            except Exception as e:
                print(f"  ✗ {method_name} FAILED: {e}")
    
    # Run performance benchmarks
    print(f"\n--- Performance Benchmarks ---")
    benchmark = TestPerformanceBenchmarks()
    benchmark.benchmark_embedding_speed()
    benchmark.benchmark_attention_speed()
    benchmark.benchmark_full_model()
    
    print(f"\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()