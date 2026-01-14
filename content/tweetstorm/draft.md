# Tweet Storm Draft: "You Can Just Do Shit"

## The Hook Options (Pick One)

### Option A: The Timeline Hook
```
Thursday: Pitched an idea. Got skeptical looks.
Friday: Said "fuck it" and built it anyway.
Tuesday: Team of 3 assigned to productionize it.

Here's how a shower thought became a production ML pipeline in 6 days:
```

### Option B: The Numbers Hook
```
40,000 mammograms. 92% accuracy. Built in one day.

Everyone said it was complicated. I said "fuck it, one day of execution."

6 days later, there's a team assigned to ship it.

Thread:
```

### Option C: The "You Can Just Do Things" Hook
```
"Why can't we just train a small model to find these?"

*skeptical looks around the room*

I chuckled. Woke up Friday and built it anyway.

92% accuracy. 100 images/second. Now a team of 3 is productionizing it.

you can just do things.
```

---

## Tweet 2: The Problem

```
The context: We're organizing ~1M mammogram images for a breast cancer research project.

Problem: DICOM metadata SHOULD tell us which images have biopsy markers or magnification views.

Reality: It depends on the hospital's SOP. And this dataset spans UK's entire hospital network.
```

[dicom_metadata_inconsistency.png] Caption: Screenshot showing inconsistent DICOM metadata fields across different sources - some populated, some empty

---

## Tweet 3: The Insight

```
In a meeting with a radiologist, I learned:

- Biopsy markers = giant white opaque tools visible in the image
- Mag views = horizontal artifacts (like old VCR tracking lines)

Both are VISUALLY distinct.

My brain: "...why can't we just train a small ResNet to spot these?"
```

[biopsy_tool_example.png] Caption: Example mammogram showing a bright biopsy marker tool (ensure fully de-identified)

[magview_example.png] Caption: Example showing the horizontal compression plate artifacts of a mag view

---

## Tweet 4: The "Fuck It" Moment

```
I pitched it. Got skeptical looks.

I'm not even a vision researcher. My bread and butter is NLP.

But I've read papers. I know the technology exists.

Friday morning I woke up and thought:

"Fuck it. 1 day of execution to satisfy my own itch."
```

---

## Tweet 5: The Build (Technical)

```
The stack:
- pydicom for DICOM normalization (VOI LUT, the proper way)
- fastai + ResNet18 (ImageNet pretrained)
- SQLite for labels (sqlite-utils)
- FastHTML for a keyboard-driven labeling UI

1 = Biopsy
2 = Mag View
3 = Neither

No frameworks. No cloud. Just Python.
```

[labeling_ui.png] Caption: Screenshot of the dark-mode labeling interface with keyboard shortcuts displayed

---

## Tweet 6: The Iteration Loop

```
First iteration: 50 labels, 67% accuracy.

"Ok not terrible for 50 images."

Surfaced random predictions. Relabeled mistakes. Retrained.

4-5 rounds later: 92% accuracy.

The whole process took an afternoon.
```

[accuracy_progression.png] Caption: Simple chart showing accuracy improving across iterations: 67% → 78% → 85% → 89% → 92%

---

## Tweet 7: The Speed

```
Then I exported to ONNX for inference.

CPU only. No GPU required.

~100 images per second.

40,000 images classified in under 5 minutes.

Not fucking bad for a Friday afternoon project.
```

[inference_speed.png] Caption: Terminal output showing batch processing speed or timing statistics

---

## Tweet 8: The Reveal

```
I almost forgot about it.

Tuesday meeting: Coworker mentions he's sifting through DICOM headers to find patterns.

I chuckle. "Oh yeah, I took a stab at that too."

Explained what I built.

My boss perked up.
```

---

## Tweet 9: The Outcome

```
Within an hour:

- Pulled in the PhD student leading the research
- Formed a 3-person team
- Tasked with productionizing the pipeline

A shower thought → skeptical looks → "fuck it, one day" → production roadmap.

6 days. Including a weekend I didn't work.
```

---

## Tweet 10: The Metrics (Show Your Work)

```
Results:
- 92% accuracy (3-class: biopsy / mag / neither)
- 100+ images/sec on CPU
- 40K images in <5 min
- Study-level aggregation (if ANY image flags, study flags)

[confusion_matrix.png] Caption: 3x3 confusion matrix showing classification performance
```

---

## Tweet 11: The False Negatives (Transparency)

```
Where does it fail?

Mostly edge cases:
- Biopsy markers partially outside frame
- Mag views with subtle artifacts
- Images with both features

Honest about limitations. That's why we're productionizing with proper validation.
```

[false_negative_example.png] Caption: Example of a misclassified image with annotation explaining why (de-identified)

---

## Tweet 12: The Philosophy

```
The takeaway isn't "I'm smart."

It's: most ideas die in the "sounds complicated" phase.

Skip that phase.

Build the shitty v1.
Let reality tell you if you're wrong.

Pessimists sound smart. Optimists build things.
```

---

## Tweet 13: The Delusional Close

```
Delusional belief in self is underrated.

Not "I know everything."
But "I can figure this out in a day."

The gap between "interesting idea" and "working prototype" is smaller than you think.

You can just do shit.
```

---

## Tweet 14: The CTA

```
If you're building AI in healthcare, or just like people who build before they're "ready"...

I'm exploring what's next. DMs open.

[Link to longer blog post with details]
```

---

## Alternative Shorter Version (7 Tweets)

### Tweet 1
```
Thursday: "Why can't we just train a model for this?"
*skeptical looks*

Friday: Built it anyway.

Tuesday: Team of 3 assigned to productionize it.

you can just do things.
```

### Tweet 2
```
The problem: Organizing 1M mammograms for breast cancer research.

Metadata SHOULD tell us which images have biopsy markers or mag views.

Reality: Data quality varies by hospital. We needed visual classification.
```

### Tweet 3
```
The build:
- fastai + ResNet18
- Keyboard-driven labeling UI (1/2/3 keys)
- 4-5 training iterations
- ONNX export for inference

Friday afternoon project.
```

[labeling_ui.png] Caption: The labeling interface

### Tweet 4
```
Results:
- 92% accuracy after 500 labels
- 100 images/sec on CPU
- 40K images in <5 min

[confusion_matrix.png] Caption: Model performance
```

### Tweet 5
```
Tuesday meeting: Casually mentioned I'd solved the problem.

Boss: *perks up*

Now there's a team productionizing it.

6 days from idea to production roadmap.
```

### Tweet 6
```
Most ideas die in the "sounds complicated" phase.

Skip that phase. Build the shitty v1.

Pessimists sound smart. Optimists build things.
```

### Tweet 7
```
Building in healthcare AI or looking for someone who ships fast?

DMs open. Exploring what's next.

Full writeup: [blog link]
```

---

## Image/Asset Checklist

| Asset | Purpose | Notes |
|-------|---------|-------|
| `labeling_ui.png` | Show the keyboard-driven dark mode UI | Most visually interesting, include keyboard shortcut hints |
| `biopsy_tool_example.png` | Show what biopsy markers look like | MUST be de-identified, bright white marker visible |
| `magview_example.png` | Show mag view horizontal artifacts | MUST be de-identified, show the VCR-like lines |
| `confusion_matrix.png` | Prove the metrics | 3x3 matrix, clearly labeled |
| `accuracy_progression.png` | Show iteration improvement | Simple line chart 67% → 92% |
| `inference_speed.png` | Show the speed | Terminal output or timing chart |
| `false_negative_example.png` | Build credibility through transparency | Annotate why it was misclassified |
| `dicom_metadata_inconsistency.png` | Optional - shows the problem | Side-by-side of populated vs empty metadata |
