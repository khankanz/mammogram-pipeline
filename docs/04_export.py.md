# ONNX Export: From Research to Production

This script converts your PyTorch model to ONNX format for deployment. Think of it as:

- **PyTorch**: Research lab equipment (flexible, heavy, needs setup)
- **ONNX**: Production appliance (fixed function, lightweight, plug and play)

## The Core Tradeoff: Dynamic vs Static

PyTorch rebuilds the computation graph on every forward pass (dynamic). ONNX freezes the graph once (static). Same tradeoff you see everywhere in CS:

| Dynamic | Static |
|---------|--------|
| PyTorch eager | ONNX graph |
| Outlines (runtime grammar) | xgrammar (precompiled) |
| Interpreted regex | Compiled DFA |
| JIT compilation | AOT compilation |

The pattern: flexibility during development, freeze for production.

---

## History Lesson: Why PyTorch Won

PyTorch was built at Facebook AI Research for researchers who iterate fast. They chose: "Make it easy to experiment, optimize later."

TensorFlow 1.x went the other way—static graphs first, define-then-run. Researchers hated it. Writing `tf.cond()` instead of `if` statements was painful.

TensorFlow 2.0 added eager execution because PyTorch was eating their lunch.

---

## Model Recreation: Why We Rebuild

```python
def create_model(n_out=2, dropout=0.4):
    "Create EfficientNet-B0 with custom head for multi-label classification."
    model = efficientnet_b0(weights=None)  # No pretrained weights—we're loading our own
    n_in = model.classifier[1].in_features  # 1280 for EfficientNet-B0
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_in, n_out))
    return model
```

When we saved with `torch.save({'model_state_dict': ...})`, we saved JUST the weights, not the architecture. This is good practice (smaller files, version-independent) but means rebuilding the model skeleton before loading weights.

The architecture must match EXACTLY: same backbone, same head, same dimensions. Mismatch → `load_state_dict()` explodes.

---

## Why model.eval() is CRITICAL

```python
def _load_model(model_path, device='cpu'):
    # ...
    model.eval()  # CRITICAL
    assert not model.training, "Model should be in eval mode!"
    return model
```

Two reasons this matters before export:

**1. Dropout**: In training, dropout randomly zeros 40% of neurons. In eval, it scales all neurons by 0.6 instead. If you export in training mode, your ONNX model will have RANDOM dropout baked in—different outputs every run!

**2. BatchNorm** (if present): Uses running statistics in eval mode; per-batch statistics in training mode. Export in training mode → your ONNX model expects the same batch size forever.

---

## ONNX Tracing: Recording the Computation

```python
def _create_dummy_input(batch_size=1, img_size=224, seed=42):
    "Create dummy input for ONNX tracing. Fixed seed for reproducibility."
    torch.manual_seed(seed)
    return torch.randn(batch_size, 3, img_size, img_size)
```

ONNX works by TRACING. PyTorch runs your model with this dummy input and RECORDS every operation. That recorded sequence becomes your ONNX graph.

Same pattern as everywhere:
- JIT compiler watches execution, learns patterns, compiles optimized version
- xgrammar precomputes all possible grammar states before generation

The dummy input must be the right SHAPE, but values don't matter (it's just to trace the ops).

Shape breakdown: `(batch, channels, height, width)`
- `1`: batch size (will be dynamic via `dynamic_axes`, but needs concrete value for tracing)
- `3`: RGB channels (what EfficientNet expects)
- `224, 224`: spatial dimensions (ImageNet standard, what our preprocessing produces)

---

## The Export Call

```python
def _export_onnx(model, dummy_input, output_path, opset=17):
    assert not model.training, "Model must be in eval mode for export!"
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['image'],       # Human-readable name
        output_names=['logits'],     # Human-readable name
        dynamic_axes={
            'image': {0: 'batch'},   # Axis 0 can vary—enables batching
            'logits': {0: 'batch'}   # Output batch axis also dynamic
        },
        opset_version=opset,         # ONNX operator version
        do_constant_folding=True,    # Precompute constant ops
        dynamo=False,                # TorchScript (stable), not TorchDynamo
        training=torch.onnx.TrainingMode.EVAL,  # EXPLICITLY eval mode
    )
```

Key parameters:

**`input_names` / `output_names`**: Human-readable names used when running inference:
```python
session.run(None, {'image': batch})  # 'image' comes from input_names
```

**`dynamic_axes`**: Without this, batch_size=1 forever. With it, you can batch inputs at inference time.

**`do_constant_folding`**: Optimization that precomputes constant operations (e.g., `x * 2.0` where `2.0` never changes).

**`dynamo=False`**: Use TorchScript (stable) not TorchDynamo (newer, can cause SLURM pain).

---

## ONNX Opset Version Explained

Think of ONNX operators as LEGO blocks: Conv, MatMul, ReLU, Sigmoid, Add, Mul, Reshape...

Different opsets have different blocks available:
- Opset 11: Basic ops
- Opset 17: Adds newer ops like Gelu, better LayerNorm

When Geohot says "my hardware supports XYZ ops," he means his chip has dedicated silicon for those operations. If your model uses an op the hardware doesn't support → software fallback (slow) or failure.

Opset 17 is safe: modern enough for EfficientNet, compatible with most deployment targets.

---

## Verification: Did Export Preserve the Function?

```python
def _verify_outputs(model, dummy_input, onnx_path, atol=1e-2):
    "Compare PyTorch vs ONNX outputs. Return max difference."
    model.eval()
    
    # Run PyTorch
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()
    
    # Run ONNX
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    onnx_out = session.run(None, {'image': dummy_input.numpy()})[0]
    
    max_diff = np.abs(torch_out - onnx_out).max()
    return max_diff, max_diff <= atol
```

Tolerance of 1e-2 (0.01) is appropriate because:

1. These are LOGITS (pre-sigmoid), so values can be large (219, -268)
2. After sigmoid: `sigmoid(219) ≈ 1.0`, `sigmoid(-268) ≈ 0.0`
3. A difference of 0.005 in logits changes the probability by ~0.00001%
4. FP32 accumulation differences across thousands of ops add up

What matters: `sigmoid(torch_out) ≈ sigmoid(onnx_out)` for classification. For logits of magnitude 200+, a diff of 0.01 is noise.

---

## Why One Dummy Input is Enough (Usually)

For DETERMINISTIC, FEEDFORWARD networks (like EfficientNet):
- No internal state that changes between inputs
- No randomness in eval mode
- Same input → same output, always

So ONE input genuinely verifies the export worked correctly.

**HOWEVER**, if your model has data-dependent control flow:

```python
if x.mean() > 0.5: 
    return self.branch_a(x)
else: 
    return self.branch_b(x)
```

Then ONNX only traces ONE path. The dummy input determines which path gets exported! For production with such models: test multiple inputs (zeros, ones, random, extreme values).

**Connection to signals & systems**: The Dirac delta (impulse) characterizes LTI systems completely. Neural networks are NONLINEAR, so no single input characterizes everything. BUT for the narrow question "did export preserve the function?" one input suffices because we're checking graph equivalence, not full system characterization.

---

## CLI Usage

```bash
# Default paths
python scripts/04_export.py

# Custom paths
python scripts/04_export.py --input models/best.pth --output models/best.onnx
```

Output:
```
Loading model from models/model.pth...
Exporting to models/model.onnx...
Verifying ONNX model structure...
Comparing PyTorch vs ONNX outputs...
  PyTorch output: [[ 219.123  -268.456]]
  ONNX output:    [[ 219.118  -268.451]]
Max output difference: 0.005432
Verification passed!

Model: EfficientNet-B0
Size: 16.2 MB
Input: (batch, 3, 224, 224)
Output: (batch, 2) - [biopsy, magview] logits

Export complete!
Next: Run inference with: python scripts/05_inference.py
```

---

## The Deployment Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ TRAINING (03_train.py)                                          │
│   PyTorch model → model.pth (weights only)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ EXPORT (04_export.py)                                           │
│   Recreate architecture + load weights                          │
│   Trace with dummy input → freeze graph                         │
│   Verify outputs match → model.onnx                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ INFERENCE (05_inference.py)                                     │
│   ONNX Runtime loads frozen graph                               │
│   No PyTorch dependency needed                                  │
│   Batch inference, hardware acceleration                        │
└─────────────────────────────────────────────────────────────────┘
```

That's the export pipeline. PyTorch for training flexibility, ONNX for production speed. Same weights, frozen graph, verified equivalence.