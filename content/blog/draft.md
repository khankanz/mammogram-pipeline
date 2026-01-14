# Blog Post Draft: "How a Shower Thought Became a Production ML Pipeline in 6 Days"

## Alternate Titles
- "You Can Just Do Shit: Building a Mammogram Classifier in a Day"
- "From Skeptical Looks to Team of 3: A Weekend ML Project"
- "The Gap Between 'Interesting Idea' and 'Working Prototype' Is Smaller Than You Think"

---

## The Hook (Choose One)

### Option A: The Timeline
> Thursday: I pitched an idea in a meeting. Got skeptical looks.
>
> Friday: I said "fuck it" and built it anyway.
>
> Tuesday: There's a team of 3 assigned to productionize it.
>
> This is the story of how a shower thought became a production ML pipeline. And why "you can just do shit" is the most underrated career advice I've ever followed.

### Option B: The Numbers
> 40,000 mammogram images. 92% classification accuracy. Built in a single day.
>
> When I mentioned the idea, I got skeptical looks around the room. Six days later, a team was assigned to ship it to production.
>
> Here's everything I built, how I built it, and what I learned about the gap between "interesting idea" and "working prototype."

---

## Part 1: The Problem

We're working on a DCIS (Ductal Carcinoma In Situ) upstaging project. Think: predicting which early-stage breast cancers will become more aggressive. The kind of research that could change treatment decisions for thousands of patients.

The dataset? Roughly **a million DICOM images** across multiple studies.

Here's where it gets complicated:

We have a UK dataset called OPTIMUM that uses a different classification system than what we use in North America. The UK uses descriptive terms (like "casting") for micro-calcifications. We use BI-RADS, a standardized lexicon. If we want to test whether our models generalize across populations, we need to map between these systems.

Spoiler: that mapping doesn't exist in a clean form.

So we had a meeting with a radiologist.

[meeting_context.png] Caption: The research context - cross-population validation requires understanding image metadata and characteristics

---

## Part 2: The Insight

During the meeting, the radiologist mentioned two visual features that are diagnostically important:

**Biopsy Tools**: If you see a bright white metallic marker in the image, it means the lesion was suspicious enough to warrant a biopsy. This is critical metadata about the case.

**Magnification Views (Mag Views)**: These show as horizontal artifacts—imagine old VCR tracking lines across the image. If a mag view was ordered, it means something was suspicious enough to warrant a closer look.

[biopsy_tool_example.png] Caption: A mammogram showing a biopsy marker—the bright white opaque tool is visually unmistakable

[magview_example.png] Caption: A magnification view showing the characteristic horizontal compression plate artifacts

Here's the thing: this information SHOULD be stored in the DICOM metadata. The standard supports it.

But reality is messier. Whether these fields are populated depends on:
- The hospital's Standard Operating Procedures
- The manufacturer of the imaging equipment
- Who was working the scanner that day

And OPTIMUM spans the UK's entire hospital network. Inconsistency is the only consistency.

[dicom_metadata_inconsistency.png] Caption: The same metadata field across different sources—sometimes populated, sometimes empty

---

## Part 3: The "Fuck It" Moment

Sitting in that meeting, my brain did the thing it does:

*"Wait. Biopsy markers are giant bright white objects. Mag views have distinct horizontal lines. These are visually obvious patterns. Why can't we just... train a small model to spot them?"*

I mentioned this out loud.

Got skeptical looks.

Fair enough. I'm not a vision researcher. My background is NLP. I've never trained an image classifier for production use.

But I've read papers. I know ResNets. I know transfer learning. I know active learning loops.

I left the meeting and mostly forgot about it.

Friday morning, I woke up with the idea still rattling around.

**"Fuck it. One day of execution. If it's a bad idea, I'll know by 5pm."**

---

## Part 4: The Build

Here's exactly what I built.

### The Stack

| Component | Choice | Why |
|-----------|--------|-----|
| DICOM Processing | pydicom + joblib | Proper VOI LUT normalization, parallel processing |
| Model | ResNet18 (ImageNet pretrained) | Small enough to train fast, big enough to learn patterns |
| Framework | fastai v2 | DataBlock API, lr_find, fine_tune—batteries included |
| Database | SQLite via sqlite-utils | Single file, observable, no server overhead |
| Labeling UI | FastHTML + HTMX | Server-rendered, keyboard shortcuts, zero JS frameworks |
| Inference | ONNX Runtime (CPU) | Fast, portable, no GPU dependencies |

### The Architecture

[architecture_diagram.png] Caption: Pipeline architecture—DICOM files → normalized thumbnails → labeling UI → training → ONNX export → batch inference

**Three mutually exclusive classes:**
1. **Biopsy Tool**: Bright marker visible
2. **Mag View**: Compression plate artifacts visible
3. **Neither**: Normal screening image

Labels are mutually exclusive at the image level. Final output aggregates to the **study level**—if ANY image in a study has a biopsy marker or mag view, the entire study gets flagged.

### The Labeling UI

[labeling_ui_full.png] Caption: The keyboard-driven labeling interface—dark background, centered image, single-keypress labeling

This was honestly the most satisfying part to build.

- Dark background (#0a0a0a) because I'm staring at medical images
- Single keypress labels AND advances: `1` = Biopsy, `2` = Mag View, `3` = Neither
- Preloads next 3 images for instant transitions
- Stats bar showing total labeled, class distribution
- `u` to undo if I fat-finger a key

No mouse clicking. No multi-step forms. Just keyboard flow.

```python
# The core labeling logic
@app.post("/label")
def label(choice: str, current_id: int):
    if choice == "biopsy":
        db.update(id=current_id, has_biopsy_tool=1, has_mag_view=0)
    elif choice == "mag":
        db.update(id=current_id, has_biopsy_tool=0, has_mag_view=1)
    else:  # neither
        db.update(id=current_id, has_biopsy_tool=0, has_mag_view=0)

    return get_next_unlabeled_image()
```

### DICOM Normalization

This is where most people get it wrong. You can't just read pixel values and normalize to 0-255.

DICOMs use Modality LUTs and VOI LUTs (Value of Interest Look-Up Tables) to map raw sensor data to diagnostically meaningful values. Skip these and your images look like garbage.

```python
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut

def normalize_dicom(pixel_array, dataset):
    # Apply Modality LUT (RescaleSlope/Intercept)
    arr = apply_modality_lut(pixel_array, dataset)

    # Apply VOI LUT (windowing)
    arr = apply_voi_lut(arr, dataset)

    # Handle MONOCHROME1 inversion
    if dataset.PhotometricInterpretation == 'MONOCHROME1':
        arr = arr.max() - arr

    # Normalize to 0-255
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
    return arr.astype(np.uint8)
```

[dicom_normalization_comparison.png] Caption: Left: raw pixel values normalized naively. Right: proper VOI LUT applied. The difference is diagnostic quality.

---

## Part 5: The Iteration Loop

Here's where the magic happened.

**Iteration 1**: Labeled 50 images. Trained. **67% accuracy.**

"Okay, not terrible for 50 images."

**Iteration 2**: Model predicts on unlabeled data. I review the uncertain ones (confidence 40-60%). Correct mistakes. Retrain. **78% accuracy.**

**Iteration 3-5**: Repeat. Focus on edge cases. Add more examples of tricky mag views.

By late Friday afternoon: **92% accuracy.**

[accuracy_progression_chart.png] Caption: Accuracy vs. number of labeled examples—the active learning curve

The active learning loop is key. You don't need to label thousands of images upfront. You:

1. Label a small seed set
2. Train
3. Let the model show you what it's uncertain about
4. Label those
5. Repeat

50 → 100 → 200 → 350 → ~500 labels total. That's it.

---

## Part 6: Inference Speed

I exported the model to ONNX for production inference.

```bash
python scripts/04_export.py --input models/model.pkl --output models/model.onnx
```

Then ran batch inference on CPU:

**~100 images per second. 40,000 images in under 5 minutes.**

[inference_timing.png] Caption: Batch processing output—throughput and timing statistics

No GPU required. Runs on a laptop.

The output is a study-level CSV:

```csv
study_id,has_biopsy_tool,has_mag_view,max_prob_biopsy,max_prob_mag,image_count
study_001,1,0,0.97,0.12,15
study_002,0,1,0.08,0.94,12
study_003,0,0,0.15,0.22,18
```

If ANY image in a study contains a biopsy marker or mag view, the study gets flagged. Conservative by design—we'd rather review false positives than miss real ones.

---

## Part 7: The Reveal

I patted myself on the back Friday evening and moved on with my weekend.

Almost forgot about it.

Tuesday meeting. One of my coworkers mentions he's been sifting through DICOM headers trying to find metadata patterns for the same problem.

I chuckled.

"Oh yeah, I took a stab at this too."

Pulled up the results. Showed the accuracy. Showed the speed.

**My boss perked up.**

Within the hour:
- The PhD student leading the research project was pulled in
- A 3-person team was formed
- We're now productionizing the pipeline

A shower thought → skeptical looks → "fuck it, one day" → production roadmap.

**6 days. Including a weekend I didn't touch it.**

---

## Part 8: Where It Fails (Honest Assessment)

I want to be transparent about limitations.

[confusion_matrix.png] Caption: The full confusion matrix—where the 8% errors come from

**False negatives happen when:**
- Biopsy markers are partially cut off at the edge of the frame
- Mag view artifacts are subtle or partially obscured
- Images have both features (model picks one)

[false_negative_example_1.png] Caption: Edge case—biopsy marker partially outside the imaging area, model classified as "Neither"

[false_negative_example_2.png] Caption: Subtle mag view artifacts that the model missed

**This is why we're productionizing with proper validation.** The weekend prototype proved the concept. The production system needs:
- Larger validation set
- Cross-hospital testing (generalization)
- Human-in-the-loop for edge cases
- Integration with the research workflow

The 92% gets us 90% of the way there. The last 10% is the real work.

---

## Part 9: What I Learned

### Technical Lessons

1. **Active learning is underrated.** 500 labels + iteration beats 5,000 labels + single training.

2. **DICOM normalization matters.** Use the LUTs. Don't roll your own windowing.

3. **Keyboard-driven UIs are fast.** I labeled ~500 images in under an hour. Mouse clicking would have taken 3x longer.

4. **ONNX is magic.** CPU inference at 100 img/sec with zero GPU dependencies.

### Career Lessons

1. **Most ideas die in the "sounds complicated" phase.** Skip that phase. Build the shitty v1.

2. **One day of execution beats one week of planning.** Especially for exploratory work.

3. **Credibility compounds.** The people who build things get asked to build more things.

4. **Delusional belief in self is underrated.** Not "I know everything." But "I can figure this out in a day."

---

## Part 10: The Philosophy

Here's my actual takeaway:

**You can just do shit.**

The gap between "interesting idea" and "working prototype" is almost always smaller than it looks. The tools exist. The tutorials exist. The compute is cheap.

What's rare is someone saying "fuck it, I'll find out if this works by 5pm."

Pessimists sound smart. Optimists make money.

I'm not saying every idea works. Mine don't. Most of them fail.

But the cost of testing an idea—really testing it, with code—is one day.

The cost of NOT testing it is never knowing.

---

## The Code

The full pipeline is ~500 lines of Python across 5 scripts:

1. `01_preprocess.py` — DICOM → normalized PNG thumbnails
2. `02_label_server.py` — FastHTML keyboard-driven labeling UI
3. `03_train.py` — fastai training with active learning loop
4. `04_export.py` — ONNX export
5. `05_inference.py` — Batch classification → study-level CSV

[If open-sourcing: GitHub link here]

---

## About Me

I'm [Name], a machine learning engineer with a background in NLP who apparently now builds computer vision systems on Friday afternoons.

I'm currently exploring what's next. If you're building in healthcare AI, or just appreciate people who ship before they're "ready," I'd love to connect.

**Email:** [email]
**LinkedIn:** [link]
**Twitter/X:** [link]

---

## Appendix: Technical Details

### Dependencies
```toml
[dependencies]
fastai = "^2.7"
pydicom = "^2.4"
sqlite-utils = "^3.35"
python-fasthtml = "^0.4"
onnxruntime = "^1.16"
joblib = "^1.3"
```

### Database Schema
```sql
CREATE TABLE labels (
    id INTEGER PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,
    study_id TEXT NOT NULL,
    thumbnail_path TEXT,
    frame_number INTEGER DEFAULT 0,
    has_biopsy_tool INTEGER,    -- NULL=unlabeled, 0=No, 1=Yes
    has_mag_view INTEGER,       -- NULL=unlabeled, 0=No, 1=Yes
    labeled_at TIMESTAMP,
    confidence_biopsy REAL,
    confidence_mag REAL,
    predicted_at TIMESTAMP
);
```

### Model Architecture
- Base: ResNet18 (ImageNet pretrained)
- Head: 3-class softmax (neither / biopsy_tool / mag_view)
- Input: 224x224 RGB (grayscale replicated to 3 channels)
- Training: fine_tune(6) with lr_find valley

### Performance
- Training: ~2 minutes per iteration on 8GB GPU
- Inference: ~100 images/second on CPU (ONNX)
- Memory: <2GB RAM for batch inference

---

*[Name] is a machine learning engineer currently exploring new opportunities. This project was built to solve a real research problem using accessible tools. If you're working on similar challenges or hiring for ML roles, reach out.*
