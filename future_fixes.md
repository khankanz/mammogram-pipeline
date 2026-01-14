# Future Fixes

## UI Improvements

- [x] Remove image preloading display - maximize image size in viewport
- [x] Image should be as large as possible on screen

## Model Architecture

- [x] Labels are NOT mutually exclusive - an image CAN have both MagView AND Biopsy Tool
- [x] Changed to single multi-label model with sigmoid outputs (BCEWithLogitsLoss)
- [x] Update labeling UI to allow selecting both options (toggle buttons)
- [x] Update database queries and training logic accordingly

## Completed Changes

- Multi-label labeling UI with toggle buttons (1=Biopsy, 2=MagView, 3=Neither, Enter=Confirm)
- Training script uses BCEWithLogitsLoss for independent binary predictions
- GradCAM visualization script (scripts/06_gradcam.py)
- ONNX export updated for .pth model format
- Inference script uses sigmoid (not softmax) for multi-label
