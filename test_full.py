import sys
sys.path.insert(0, 'python')
import needle as ndl
import numpy as np

print("=" * 60)
print("Needle-SSM Full Test")
print("=" * 60)

# Test 1: Basic imports
print("\n✓ Test 1: Imports")
print(f"  Backend: {ndl.backend_selection.BACKEND}")
print(f"  Available devices: {ndl.all_devices()}")

# Test 2: CPU tensor operations
print("\n✓ Test 2: CPU Tensor Operations")
x_cpu = ndl.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
y_cpu = ndl.Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
z_cpu = x_cpu + y_cpu
print(f"  CPU addition: {x_cpu.numpy()} + {y_cpu.numpy()} = {z_cpu.numpy()}")

# Test 3: CUDA tensor operations (if available)
print("\n✓ Test 3: CUDA Tensor Operations")
if ndl.cuda().enabled():
    try:
        x_gpu = ndl.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), device=ndl.cuda())
        y_gpu = ndl.Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), device=ndl.cuda())
        z_gpu = x_gpu + y_gpu
        print(f"  CUDA addition successful!")
        print(f"  Result matches CPU: {np.allclose(z_cpu.numpy(), z_gpu.numpy())}")
    except Exception as e:
        print(f"  ⚠ CUDA operation failed: {e}")
else:
    print("  ⚠ CUDA backend not available")

# Test 4: Neural network modules
print("\n✓ Test 4: Neural Network Modules")
try:
    import needle.nn as nn
    linear = nn.Linear(10, 5)
    print(f"  Linear layer created: {linear}")
except Exception as e:
    print(f"  ⚠ NN module test failed: {e}")

# Test 5: Operations
print("\n✓ Test 5: Operations")
try:
    a = ndl.Tensor(np.array([1.0, 2.0, 3.0]))
    b = ndl.ops.exp(a)
    print(f"  exp([1, 2, 3]) = {b.numpy()}")
except Exception as e:
    print(f"  ⚠ Operations test failed: {e}")

print("\n" + "=" * 60)
print("🎉 All tests completed!")
print("=" * 60)
