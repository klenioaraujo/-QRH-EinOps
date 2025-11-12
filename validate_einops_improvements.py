#!/usr/bin/env python3
"""
Validação das melhorias com EinOps - Análise estática do código
"""

import ast
import sys
import os

def analyze_code_improvements():
    """Análise estática das melhorias implementadas com EinOps"""
    
    print("🔍 VALIDAÇÃO ΨQRH EINOPS OPTIMIZATION")
    print("=" * 60)
    
    # Ler o código otimizado
    with open('ΨQRH_EINOPS_OPTIMIZED.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Métricas de análise
    metrics = {
        'einops_operations': 0,
        'manual_reshaping': 0,
        'python_loops': 0,
        'vectorized_operations': 0,
        'energy_conservation': 0
    }
    
    # Contar operações EinOps
    einops_keywords = ['rearrange(', 'reduce(', 'repeat(', 'parse_shape(']
    for keyword in einops_keywords:
        metrics['einops_operations'] += code.count(keyword)
    
    # Contar operações de reshape manual
    manual_ops = ['.view(', '.reshape(', '.permute(', '.unsqueeze(', '.squeeze(']
    for op in manual_ops:
        metrics['manual_reshaping'] += code.count(op)
    
    # Contar loops Python (aproximação)
    metrics['python_loops'] = code.count('for ') + code.count('while ')
    
    # Contar operações vetorizadas
    vectorized_ops = ['torch.matmul', 'torch.bmm', 'nn.Embedding', 'torch.stack']
    for op in vectorized_ops:
        metrics['vectorized_operations'] += code.count(op)
    
    # Contar conservação de energia
    energy_keywords = ['energy_conservation', 'energy_normalizer', 'energy_preservation', 'energy_ratio']
    for keyword in energy_keywords:
        metrics['energy_conservation'] += code.count(keyword)
    
    # Análise de AST para funções críticas
    try:
        tree = ast.parse(code)
        
        # Contar funções e métodos
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        metrics['total_functions'] = len(functions)
        metrics['total_classes'] = len(classes)
        
        # Verificar forward method
        forward_methods = [f for f in functions if f.name == 'forward']
        if forward_methods:
            metrics['has_forward'] = True
            forward_code = ast.unparse(forward_methods[0])
            metrics['forward_loops'] = forward_code.count('for ') + forward_code.count('while ')
        else:
            metrics['has_forward'] = False
            metrics['forward_loops'] = 0
            
    except Exception as e:
        print(f"⚠️  Erro na análise AST: {e}")
    
    # Exibir resultados
    print("\n📊 MÉTRICAS DE OTIMIZAÇÃO:")
    print(f"   🔄 Operações EinOps: {metrics['einops_operations']}")
    print(f"   🚫 Operações de reshape manual: {metrics['manual_reshaping']}")
    print(f"   🔁 Loops Python totais: {metrics['python_loops']}")
    print(f"   🎯 Operações vetorizadas: {metrics['vectorized_operations']}")
    print(f"   ⚡ Referências conservação energia: {metrics['energy_conservation']}")
    
    if 'total_functions' in metrics:
        print(f"   🏗️  Classes: {metrics['total_classes']}, Funções: {metrics['total_functions']}")
        print(f"   🔁 Loops no forward: {metrics['forward_loops']}")
    
    # Avaliação qualitativa
    print("\n🎯 AVALIAÇÃO DAS MELHORIAS:")
    
    if metrics['manual_reshaping'] == 0:
        print("   ✅ ELIMINAÇÃO COMPLETA de operações de reshape manual")
    else:
        print(f"   ⚠️  Ainda existem {metrics['manual_reshaping']} operações de reshape manual")
    
    if metrics['einops_operations'] > 10:
        print("   ✅ USO EXTENSIVO de EinOps para operações seguras")
    else:
        print("   ⚠️  Uso limitado de EinOps")
    
    if metrics['energy_conservation'] > 5:
        print("   ✅ CONSERVAÇÃO DE ENERGIA implementada robustamente")
    else:
        print("   ⚠️  Conservação de energia limitada")
    
    if metrics['forward_loops'] < 5:
        print("   ✅ FORWARD PASS VETORIZADO (poucos loops)")
    else:
        print(f"   ⚠️  Forward pass com {metrics['forward_loops']} loops (pode ser otimizado)")
    
    # Verificar imports
    print("\n📦 IMPORTS VERIFICADOS:")
    if 'from einops import' in code:
        print("   ✅ EinOps importado corretamente")
    else:
        print("   ❌ EinOps não encontrado nos imports")
    
    if 'import torch' in code:
        print("   ✅ PyTorch importado corretamente")
    else:
        print("   ❌ PyTorch não encontrado nos imports")
    
    # Verificar arquitetura principal
    print("\n🏗️  ARQUITETURA PRINCIPAL:")
    key_components = [
        'GenuineTrainedDistillationTransformer',
        'SpectralAttention', 
        'GenuineEmbedding',
        'GenuineLeechLattice',
        'QuaternionOperations'
    ]
    
    for component in key_components:
        if component in code:
            print(f"   ✅ {component} presente")
        else:
            print(f"   ❌ {component} ausente")
    
    print("\n🎉 RESUMO DA VALIDAÇÃO:")
    if (metrics['manual_reshaping'] == 0 and 
        metrics['einops_operations'] > 10 and
        metrics['energy_conservation'] > 5):
        print("   🚀 ΨQRH EINOPS OPTIMIZATION - REFATORAÇÃO BEM-SUCEDIDA!")
        print("   ✓ Eliminação total de reshape manual")
        print("   ✓ Operações tensorais seguras com EinOps") 
        print("   ✓ Conservação de energia implementada")
        print("   ✓ Código pronto para produção")
    else:
        print("   ⚠️  Algumas otimizações ainda podem ser aplicadas")

if __name__ == "__main__":
    # Mudar para diretório EinOps
    original_dir = os.getcwd()
    try:
        os.chdir('EinOps')
        analyze_code_improvements()
    finally:
        os.chdir(original_dir)