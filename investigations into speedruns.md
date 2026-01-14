# Investigations into Speedruns

## Inference Speed Summary

| Metric | Value |
|--------|-------|
| Total time | 75.05s |
| Total frames | 592 |
| **Throughput** | **7.9 frames/sec** |

## Breakdown

| Phase | Time | % of Total |
|-------|------|------------|
| DICOM discovery | 0.01s | <1% |
| DICOM loading | 69.49s | 93% |
| Preprocessing | 2.52s | 3% |
| Inference | 2.42s | 3% |
| DB operations | 0.48s | 1% |

## Analysis

The bottleneck is **DICOM loading** (93% of total time). The actual model inference is only 2.4 seconds for 592 images.

### Potential Optimizations

1. **Parallel DICOM loading** - Use multiprocessing to load multiple DICOMs simultaneously
2. **Pre-convert to PNG/JPG** - Convert DICOMs to standard image formats once, then use those for inference
3. **Memory-mapped files** - Cache decoded pixel data
4. **GPU-accelerated decoding** - Use nvJPEG or similar for image decoding
5. **Batch file I/O** - Read multiple files in parallel using async I/O

### Hardware

- Running on CPU for ONNX inference (CUDA provider not available due to missing cuDNN)
- With GPU acceleration, inference time could be reduced further
