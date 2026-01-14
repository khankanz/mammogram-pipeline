# Images Needed: Blog Post

## Required Assets (In Order of Appearance)

### 1. `meeting_context.png` (Optional)
**Location:** Part 1 - The Problem
**Purpose:** Set the research context
**Options:**
- Simple diagram showing: UK Dataset (OPTIMUM) ↔ Mapping Problem ↔ North America (BI-RADS)
- Or skip this—the text explains it well enough

---

### 2. `biopsy_tool_example.png` ⭐
**Location:** Part 2 - The Insight
**Caption:** "A mammogram showing a biopsy marker—the bright white opaque tool is visually unmistakable"
**Requirements:**
- De-identified mammogram with visible biopsy marker
- Could add subtle annotation (arrow, circle) pointing to marker
- High contrast so marker is obvious even at thumbnail size

**Privacy:** See privacy checklist below. Consider using a public dataset or synthetic example.

---

### 3. `magview_example.png` ⭐
**Location:** Part 2 - The Insight
**Caption:** "A magnification view showing the characteristic horizontal compression plate artifacts"
**Requirements:**
- De-identified mammogram showing horizontal line artifacts
- VCR tracking line analogy should be obvious

---

### 4. `dicom_metadata_inconsistency.png`
**Location:** Part 2 - The Insight
**Caption:** "The same metadata field across different sources—sometimes populated, sometimes empty"
**What to show:**
```
┌───────────────────────────────────────────────────────────────┐
│ Hospital A (UK)               │ Hospital B (UK)              │
├───────────────────────────────┼──────────────────────────────┤
│ ViewPosition: MLO             │ ViewPosition:                │
│ Manufacturer: GE Healthcare   │ Manufacturer: Hologic        │
│ BodyPartExamined: BREAST      │ BodyPartExamined:            │
│ ImageType: ['ORIGINAL']       │ ImageType: ['DERIVED']       │
└───────────────────────────────┴──────────────────────────────┘
```
Side-by-side comparison emphasizing inconsistency.

---

### 5. `architecture_diagram.png` ⭐
**Location:** Part 4 - The Build
**Caption:** "Pipeline architecture—DICOM files → normalized thumbnails → labeling UI → training → ONNX export → batch inference"
**Style:** C4 Context level or simple flow diagram
**Components:**
```
┌─────────┐    ┌─────────────┐    ┌───────────┐    ┌─────────┐
│ DICOMs  │───▶│ Preprocess  │───▶│ Thumbnails│───▶│ SQLite  │
└─────────┘    │ (pydicom)   │    │ (224x224) │    │   DB    │
               └─────────────┘    └───────────┘    └────┬────┘
                                                        │
┌─────────┐    ┌─────────────┐    ┌───────────┐    ┌────▼────┐
│  CSV    │◀───│  Inference  │◀───│   ONNX    │◀───│ FastHTML│
│ Output  │    │  (batch)    │    │  Export   │    │ Labeler │
└─────────┘    └─────────────┘    └───────────┘    └─────────┘
```

**Tools:** Draw.io, Excalidraw, or even ASCII art styled nicely

---

### 6. `labeling_ui_full.png` ⭐⭐ HERO IMAGE
**Location:** Part 4 - The Build (The Labeling UI section)
**Caption:** "The keyboard-driven labeling interface—dark background, centered image, single-keypress labeling"
**What to capture:**
- Full interface screenshot
- Dark mode visible
- A mammogram displayed (de-identified)
- Stats bar: "Labeled: 247 | Biopsy: 45 | Mag: 38 | Neither: 164"
- Keyboard hints: "1=Biopsy  2=Mag  3=Neither  s=Skip  u=Undo"
- Current study/filename visible

**This is your signature image.** Make it look polished.

---

### 7. `dicom_normalization_comparison.png`
**Location:** Part 4 - DICOM Normalization section
**Caption:** "Left: raw pixel values normalized naively. Right: proper VOI LUT applied. The difference is diagnostic quality."
**Layout:** Side-by-side comparison
```
┌────────────────────┐  ┌────────────────────┐
│   NAIVE NORMALIZE  │  │   PROPER VOI LUT   │
│                    │  │                    │
│   (washed out,     │  │   (proper contrast │
│    no contrast)    │  │    clinical grade) │
└────────────────────┘  └────────────────────┘
```

---

### 8. `accuracy_progression_chart.png` ⭐
**Location:** Part 5 - The Iteration Loop
**Caption:** "Accuracy vs. number of labeled examples—the active learning curve"
**Type:** Line chart with markers
**Data points:**
| Iteration | Labels | Accuracy |
|-----------|--------|----------|
| 1 | 50 | 67% |
| 2 | 100 | 78% |
| 3 | 200 | 85% |
| 4 | 350 | 89% |
| 5 | 500 | 92% |

**Code to generate:**
```python
import matplotlib.pyplot as plt

iterations = [1, 2, 3, 4, 5]
labels = [50, 100, 200, 350, 500]
accuracy = [67, 78, 85, 89, 92]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(iterations, accuracy, 'o-', linewidth=2, markersize=10, color='#2563eb')
ax.set_xlabel('Training Iteration', fontsize=12)
ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax.set_ylim(60, 100)
ax.set_xticks(iterations)
ax.grid(True, alpha=0.3)

# Add label counts as secondary x-axis labels
for i, (it, lab, acc) in enumerate(zip(iterations, labels, accuracy)):
    ax.annotate(f'{acc}%\n({lab} labels)',
                xy=(it, acc), xytext=(0, 15),
                textcoords='offset points', ha='center', fontsize=10)

ax.set_title('Active Learning: Accuracy Improves with Each Iteration', fontsize=14)
plt.tight_layout()
plt.savefig('accuracy_progression_chart.png', dpi=150)
```

---

### 9. `inference_timing.png`
**Location:** Part 6 - Inference Speed
**Caption:** "Batch processing output—throughput and timing statistics"
**What to show:** Terminal output or formatted stats card
```
$ python scripts/05_inference.py --input-dir data/dicoms --output results.csv

Processing 40,000 images...
████████████████████████████████████████ 100%

Inference Complete
──────────────────────────────────────────
Total images:     40,000
Total time:       4m 23s
Throughput:       152 images/sec
Model:            model.onnx (CPU)
Output:           results.csv

Study-level results: 2,847 studies
  - With biopsy marker: 423 studies (14.9%)
  - With mag view: 312 studies (11.0%)
  - Neither: 2,112 studies (74.1%)
```

---

### 10. `confusion_matrix.png` ⭐⭐ PROOF IMAGE
**Location:** Part 8 - Where It Fails
**Caption:** "The full confusion matrix—where the 8% errors come from"
**Requirements:**
- 3x3 matrix: Neither / Biopsy Tool / Mag View
- Show actual counts AND percentages
- Color-coded (seaborn Blues or similar)
- Overall accuracy visible: "92% Accuracy"

---

### 11. `false_negative_example_1.png`
**Location:** Part 8 - Where It Fails
**Caption:** "Edge case—biopsy marker partially outside the imaging area, model classified as 'Neither'"
**Requirements:**
- De-identified image
- Annotation showing where the marker is (barely visible/cut off)
- Text overlay: "Predicted: Neither | Actual: Biopsy Tool"

---

### 12. `false_negative_example_2.png`
**Location:** Part 8 - Where It Fails
**Caption:** "Subtle mag view artifacts that the model missed"
**Requirements:**
- De-identified image
- Annotation pointing to subtle horizontal lines
- Text overlay: "Predicted: Neither | Actual: Mag View"

---

## Asset Summary Table

| # | Filename | Priority | Section | Type |
|---|----------|----------|---------|------|
| 1 | biopsy_tool_example.png | ⭐⭐⭐ | Part 2 | Medical image |
| 2 | magview_example.png | ⭐⭐⭐ | Part 2 | Medical image |
| 3 | dicom_metadata_inconsistency.png | ⭐⭐ | Part 2 | Comparison |
| 4 | architecture_diagram.png | ⭐⭐ | Part 4 | Diagram |
| 5 | labeling_ui_full.png | ⭐⭐⭐ | Part 4 | Screenshot |
| 6 | dicom_normalization_comparison.png | ⭐⭐ | Part 4 | Comparison |
| 7 | accuracy_progression_chart.png | ⭐⭐⭐ | Part 5 | Chart |
| 8 | inference_timing.png | ⭐⭐ | Part 6 | Terminal/Stats |
| 9 | confusion_matrix.png | ⭐⭐⭐ | Part 8 | Chart |
| 10 | false_negative_example_1.png | ⭐⭐ | Part 8 | Annotated image |
| 11 | false_negative_example_2.png | ⭐⭐ | Part 8 | Annotated image |

---

## Privacy Checklist (Medical Images)

Before including ANY mammogram or medical image:

### Must Remove:
- [ ] Patient name
- [ ] Date of birth
- [ ] Medical record number
- [ ] Study date/time
- [ ] Institution name
- [ ] Referring physician
- [ ] Any burned-in annotations with PHI

### Best Practices:
- [ ] Crop to region of interest only
- [ ] Remove DICOM headers before sharing
- [ ] Consider using public datasets (e.g., CBIS-DDSM) for examples
- [ ] Check with IRB/privacy office if using institutional data
- [ ] When in doubt, use diagrams/illustrations instead

### Alternatives to Real Images:
1. **Use public datasets**: CBIS-DDSM, INbreast, VinDr-Mammo
2. **Use illustrations**: Draw simplified representations
3. **Use synthetic examples**: AI-generated mammogram-like images
4. **Describe instead of show**: "Biopsy markers appear as bright white metallic objects"

---

## Design Consistency Notes

For a polished look across all assets:

- **Color palette**: Blue (#2563eb), dark gray (#0a0a0a), white
- **Font**: System default or Inter/Roboto for diagrams
- **Chart style**: Minimal, clean, no 3D effects
- **Annotations**: Use arrows or circles in highlight color
- **Screenshots**: Use 2x DPI if possible, crop tightly

---

## Tools Recommendations

| Tool | Best For |
|------|----------|
| matplotlib/seaborn | Confusion matrix, accuracy chart |
| Excalidraw | Architecture diagram (hand-drawn style) |
| Draw.io | Architecture diagram (clean/professional) |
| Screenshot + Preview | Labeling UI |
| iTerm2/terminal | Inference timing |
| Figma/Canva | Polished comparison layouts |
