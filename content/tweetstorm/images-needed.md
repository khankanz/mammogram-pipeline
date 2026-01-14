# Images Needed: Tweet Storm

## Required Assets (Priority Order)

### 1. `labeling_ui.png` ⭐ MOST IMPORTANT
**Purpose:** The visual hook—shows you built a real tool
**What to capture:**
- Full screenshot of the labeling interface
- Dark background (#0a0a0a)
- A mammogram image centered
- Stats bar visible (labeled count, class distribution)
- Keyboard shortcut hints visible (1/2/3)

**Why it works:** People share tools. This looks like a real product, not a Jupyter notebook.

---

### 2. `confusion_matrix.png` ⭐ PROOF
**Purpose:** The receipts—proves the 92% claim
**What to capture:**
- 3x3 matrix with clear labels: `Neither`, `Biopsy Tool`, `Mag View`
- Actual numbers, not just percentages
- Color-coded (green diagonal, red off-diagonal)
- Include overall accuracy somewhere visible

**Format suggestion:** Use seaborn heatmap or matplotlib with annotations

```python
# Example code to generate
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Neither', 'Biopsy', 'MagView'],
            yticklabels=['Neither', 'Biopsy', 'MagView'],
            cmap='Blues')
plt.title('Classification Results: 92% Accuracy')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
```

---

### 3. `biopsy_tool_example.png` ⭐ SHOW THE PATTERN
**Purpose:** Let people see what biopsy markers look like
**Requirements:**
- **MUST be fully de-identified** (no patient info visible)
- Bright white marker clearly visible
- Could add an arrow or circle annotation pointing to it
- Caption: "Biopsy marker—bright metallic tool visible in mammogram"

**Privacy note:** Crop tightly to the region of interest. Remove any text overlays. Consider using a public dataset example.

---

### 4. `magview_example.png` ⭐ SHOW THE OTHER PATTERN
**Purpose:** Let people see the mag view artifacts
**Requirements:**
- **MUST be fully de-identified**
- Horizontal compression plate artifacts visible
- Caption: "Magnification view—horizontal artifacts from compression plates"

**Privacy note:** Same as above—be very careful with medical images.

---

### 5. `accuracy_progression.png`
**Purpose:** Shows the active learning improvement
**What to capture:**
- Simple line chart or bar chart
- X-axis: Iteration (1, 2, 3, 4, 5) or Labels (50, 100, 200, 350, 500)
- Y-axis: Accuracy (%)
- Points: 67% → 78% → 85% → 89% → 92%

**Format:** Keep it simple. Even a text-based version works:
```
Iteration 1 (50 labels):  ████████████ 67%
Iteration 2 (100 labels): ██████████████ 78%
Iteration 3 (200 labels): ████████████████ 85%
Iteration 4 (350 labels): █████████████████ 89%
Iteration 5 (500 labels): ██████████████████ 92%
```

---

### 6. `inference_speed.png`
**Purpose:** Shows the speed claim is real
**What to capture:**
- Terminal output from running batch inference
- Show throughput: "Processing: 100.3 images/sec"
- Show total time: "40,000 images in 4m 23s"

**Alternative:** A simple stats card:
```
┌─────────────────────────────┐
│ Inference Stats             │
├─────────────────────────────┤
│ Images processed: 40,000    │
│ Time elapsed: 4m 23s        │
│ Throughput: 152 img/sec     │
│ Model: ONNX (CPU only)      │
└─────────────────────────────┘
```

---

## Optional Assets

### 7. `false_negative_example.png`
**Purpose:** Build credibility through transparency
**What to capture:**
- An image the model got wrong
- Annotation explaining why (e.g., "marker at edge of frame")
- **MUST be de-identified**

---

### 8. `dicom_metadata_inconsistency.png`
**Purpose:** Shows why this problem exists
**What to capture:**
- Side-by-side comparison of DICOM headers
- One with populated manufacturer/view fields
- One with empty or inconsistent fields
- Caption: "Same field, different hospitals—metadata quality varies"

---

## Image Specifications

| Asset | Dimensions | Format | File Size Target |
|-------|------------|--------|------------------|
| labeling_ui.png | 1200x800 or 16:9 | PNG | <500KB |
| confusion_matrix.png | 800x600 | PNG | <200KB |
| biopsy_tool_example.png | 600x600 | PNG | <300KB |
| magview_example.png | 600x600 | PNG | <300KB |
| accuracy_progression.png | 800x400 | PNG | <100KB |
| inference_speed.png | 800x300 | PNG | <100KB |

---

## Privacy Checklist

Before posting ANY medical image:

- [ ] Is all patient identifying information removed?
- [ ] Are DICOM headers stripped?
- [ ] Is the image cropped to remove institutional watermarks?
- [ ] Would you be comfortable if this appeared in a newspaper?
- [ ] Have you checked with your institution's IRB/privacy office if needed?

**When in doubt, don't post the medical image.** The labeling UI, confusion matrix, and speed stats are sufficient to make the point.
