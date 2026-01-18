#! /usr/bin/env python3
"""Review UI: Validate model predictions rapidly.

REVIEW UI vs LABEL UI
─────────────────────
Label UI:  "What IS this?"      → you decide from scratch (1/2/3/4)
Review UI: "Is model RIGHT?"    → you validate (Y/N)

Label UI uses active learning (shows uncertain images).
Review UI uses random selection (representative sample).

Use Review UI when model is decent (80%+) — pressing Y most of the time
is faster than deciding from scratch.

WHAT GETS SHOWN
───────────────
Images with confidence scores that are still unlabeled.

After each training run, `update_predictions()` OVERWRITES confidence
on all unlabeled images with the latest model's predictions. So confidence
always reflects the most recent model — no stale predictions.

Labeled images have confidence=NULL (skipped by update_predictions).

ACTIVE LEARNING LOOP
────────────────────
  Train → overwrites confidence on unlabeled → Label/Review UI → repeat

Each iteration, the model tells you what it's confused about (via confidence).
Your labels make it smarter. Next iteration has different uncertainties.
"""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fasthtml.common import *
from fastcore.foundation import L

from lib.config import DB_PATH, UNDO_STACK_SIZE
from lib.db import get_db, get_image_by_id, set_labels

# =============================================================================
# CONFIG
# =============================================================================

REVIEW_PORT = 5002
QUEUE_SIZE = 100
THRESHOLD = 0.5

# =============================================================================
# STYLES
# =============================================================================

_styles = Style("""
    :root { --bg: #0a0a0a; --fg: #e0e0e0; --accent: #4a9eff; --correct: #22c55e; --wrong: #ef4444; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--fg); font-family: system-ui; height: 100vh; overflow: hidden; }
    .container { height: 100vh; display: flex; flex-direction: column; padding: 6px; }
    .stats { background: #1a1a1a; padding: 4px 10px; border-radius: 4px; display: flex; gap: 15px; font-size: 11px; }
    .stat-value { color: var(--accent); font-weight: bold; }
    .image-container { flex: 1; display: flex; justify-content: center; align-items: center; background: #111; border-radius: 4px; margin: 6px 0; min-height: 0; }
    .image-container img { max-height: 100%; max-width: 100%; object-fit: contain; }
    .filename { text-align: center; color: #666; font-size: 10px; font-family: monospace; }
    .prediction { text-align: center; padding: 8px; font-size: 14px; }
    .pred-label { display: inline-block; padding: 4px 12px; border-radius: 4px; margin: 0 4px; }
    .pred-yes { background: #166534; }
    .pred-no { background: #374151; }
    .pred-prob { color: #888; font-size: 11px; }
    .buttons { display: flex; justify-content: center; gap: 8px; padding: 6px 0; }
    .btn { padding: 8px 16px; font-size: 13px; border: none; border-radius: 4px; cursor: pointer; font-weight: 600; color: white; }
    .btn-correct { background: var(--correct); }
    .btn-wrong { background: var(--wrong); }
    .btn-skip { background: #6b7280; }
    .btn-undo { background: #7c3aed; }
    .correction-panel { background: #1a1a1a; padding: 10px; border-radius: 4px; margin: 6px 0; }
    .correction-title { font-size: 12px; color: #888; margin-bottom: 8px; }
    .kbd { background: #333; padding: 1px 5px; border-radius: 3px; font-family: monospace; margin-right: 3px; font-size: 10px; }
    .empty { text-align: center; padding: 60px; color: #666; }
""")

# =============================================================================
# KEYBOARD — two modes: main (Y/N/S/U) and correction (1/2/3/4/Esc)
# =============================================================================

def _keyboard_script(image_id, show_correction):
    "Generate keyboard handler for current mode"
    if not image_id: return Script("")
    
    if show_correction:
        return Script(f"""
            document.addEventListener('keydown', function handler(e) {{
                const id = {image_id};
                const routes = {{'1': '/correct?id='+id+'&biopsy=1&mag=0', '2': '/correct?id='+id+'&biopsy=0&mag=1',
                                 '3': '/correct?id='+id+'&biopsy=0&mag=0', '4': '/correct?id='+id+'&biopsy=1&mag=1',
                                 'Escape': '/cancel'}};
                if (routes[e.key]) {{ e.preventDefault(); document.removeEventListener('keydown', handler);
                    htmx.ajax('POST', routes[e.key], {{target: '#main', swap: 'innerHTML'}}); }}
            }});
        """)
    
    return Script(f"""
        document.addEventListener('keydown', function handler(e) {{
            const id = {image_id};
            const routes = {{'y': '/confirm?id='+id, 'n': '/show-correction', 's': '/skip', 'u': '/undo'}};
            if (routes[e.key.toLowerCase()]) {{ e.preventDefault(); document.removeEventListener('keydown', handler);
                htmx.ajax('POST', routes[e.key.toLowerCase()], {{target: '#main', swap: 'innerHTML'}}); }}
        }});
    """)

# =============================================================================
# STATE — module-level, fine for single-user tool
# =============================================================================

class State:
    def __init__(self):
        self.queue = []           # image IDs to review
        self.idx = 0              # current position
        self.undo_stack = []
        self.corrections = 0      # you disagreed with model
        self.confirmations = 0    # you agreed with model
        self.show_correction = False
    
    def reset(self): self.__init__()
    @property
    def total(self): return self.corrections + self.confirmations
    @property
    def remaining(self): return max(0, len(self.queue) - self.idx)
    @property
    def current_id(self): return self.queue[self.idx] if self.idx < len(self.queue) else None

state = State()

# =============================================================================
# DB QUERIES
# =============================================================================

def get_review_queue(db, limit=QUEUE_SIZE):
    """Get unlabeled images with predictions, random order.
    
    Why random (not active learning)?
    Review UI is for validation, not efficient labeling. Want representative sample.
    """
    rows = list(db["labels"].rows_where(
        "confidence_biopsy IS NOT NULL AND confidence_mag IS NOT NULL "
        "AND has_biopsy_tool IS NULL AND has_mag_view IS NULL",
        order_by="RANDOM()", limit=limit
    ))
    return L(rows).map(lambda r: r["id"])

def get_current_image(db):
    if state.current_id is None: return None
    return get_image_by_id(db, state.current_id)

# =============================================================================
# UI COMPONENTS
# =============================================================================

def stats_bar():
    return Div(
        Span("Queue: ", Span(f"{len(state.queue)}", cls="stat-value")),
        Span("Reviewed: ", Span(f"{state.total}", cls="stat-value")),
        Span("Confirmed: ", Span(f"{state.confirmations}", cls="stat-value")),
        Span("Corrected: ", Span(f"{state.corrections}", cls="stat-value")),
        Span("Remaining: ", Span(f"{state.remaining}", cls="stat-value")),
        cls="stats"
    )

def prediction_display(image):
    "Show model's prediction — the key difference from Label UI"
    if not image: return Div()
    
    bp, mp = image.get("confidence_biopsy") or 0, image.get("confidence_mag") or 0
    bpred, mpred = bp >= THRESHOLD, mp >= THRESHOLD
    
    return Div(
        Span("Model says: "),
        Span(f"Biopsy: {'YES' if bpred else 'NO'}", cls=f"pred-label {'pred-yes' if bpred else 'pred-no'}"),
        Span(f"({bp:.0%})", cls="pred-prob"), Span(" "),
        Span(f"MagView: {'YES' if mpred else 'NO'}", cls=f"pred-label {'pred-yes' if mpred else 'pred-no'}"),
        Span(f"({mp:.0%})", cls="pred-prob"),
        cls="prediction"
    )

def correction_panel(image_id):
    "Appears when you press N — select the correct label"
    return Div(
        Div("Model was wrong. Correct label:", cls="correction-title"),
        Div(
            Button(Span("1", cls="kbd"), "Biopsy", hx_post=f"/correct?id={image_id}&biopsy=1&mag=0", hx_target="#main", cls="btn", style="background:#2563eb;"),
            Button(Span("2", cls="kbd"), "MagView", hx_post=f"/correct?id={image_id}&biopsy=0&mag=1", hx_target="#main", cls="btn", style="background:#7c3aed;"),
            Button(Span("3", cls="kbd"), "Neither", hx_post=f"/correct?id={image_id}&biopsy=0&mag=0", hx_target="#main", cls="btn", style="background:#374151;"),
            Button(Span("4", cls="kbd"), "Both", hx_post=f"/correct?id={image_id}&biopsy=1&mag=1", hx_target="#main", cls="btn", style="background:#0891b2;"),
            Button("Cancel", hx_post="/cancel", hx_target="#main", cls="btn btn-skip"),
            cls="buttons"
        ),
        cls="correction-panel"
    )

def main_buttons(image_id):
    if not image_id: return Div()
    if state.show_correction: return correction_panel(image_id)
    
    return Div(
        Button(Span("Y", cls="kbd"), "Correct", hx_post=f"/confirm?id={image_id}", hx_target="#main", cls="btn btn-correct"),
        Button(Span("N", cls="kbd"), "Wrong", hx_post="/show-correction", hx_target="#main", cls="btn btn-wrong"),
        Button(Span("S", cls="kbd"), "Skip", hx_post="/skip", hx_target="#main", cls="btn btn-skip"),
        Button(Span("U", cls="kbd"), "Undo", hx_post="/undo", hx_target="#main", cls="btn btn-undo"),
        cls="buttons"
    )

def main_content(db):
    if not state.queue:
        state.queue = get_review_queue(db)
        state.idx = 0
    
    image = get_current_image(db)
    image_id = image["id"] if image else None
    
    if not image:
        acc = state.confirmations / max(1, state.total)
        return Div(
            stats_bar(),
            Div(H2("Review complete!"),
                P(f"Confirmed: {state.confirmations} | Corrected: {state.corrections}"),
                P(f"Model accuracy on reviewed: {acc:.0%}"),
                P(Button("Reset Queue", hx_post="/reset", hx_target="#main", cls="btn btn-skip")),
                cls="empty"),
            id="main"
        )
    
    filename = image["filename"].split("/")[-1][:50]
    return Div(
        stats_bar(),
        Div(Img(src=f"/image/{image_id}", alt=filename), cls="image-container"),
        prediction_display(image),
        Div(filename, cls="filename"),
        main_buttons(image_id),
        _keyboard_script(image_id, state.show_correction),
        id="main"
    )

# =============================================================================
# APP
# =============================================================================

app, rt = fast_app(hdrs=(_styles,), secret_key=os.urandom(24))

# =============================================================================
# ROUTES
# =============================================================================

@app.get("/")
def homepage():
    db = get_db(DB_PATH)
    return Title("Review Predictions"), Div(main_content(db), cls="container")

@app.get("/image/{image_id}")
def serve_thumbnail(image_id: int):
    db = get_db(DB_PATH)
    image = get_image_by_id(db, image_id)
    if not image: return Response("Not found", status_code=404)
    thumb_path = Path(image["thumbnail_path"])
    if not thumb_path.exists(): return Response("Thumbnail not found", status_code=404)
    return FileResponse(thumb_path, media_type="image/png")

@app.post("/confirm")
def confirm_prediction(id: int):
    "Model was right — use its prediction as label"
    db = get_db(DB_PATH)
    image = get_image_by_id(db, id)
    
    if image:
        state.undo_stack.append({"idx": state.idx, "action": "confirm", "id": id})
        if len(state.undo_stack) > UNDO_STACK_SIZE: state.undo_stack.pop(0)
        
        biopsy = 1 if (image.get("confidence_biopsy") or 0) >= THRESHOLD else 0
        mag = 1 if (image.get("confidence_mag") or 0) >= THRESHOLD else 0
        set_labels(db, id, biopsy, mag)
        state.confirmations += 1
    
    state.idx += 1
    state.show_correction = False
    return main_content(db)

@app.post("/show-correction")
def show_correction_panel():
    db = get_db(DB_PATH)
    state.show_correction = True
    return main_content(db)

@app.post("/cancel")
def cancel_correction():
    db = get_db(DB_PATH)
    state.show_correction = False
    return main_content(db)

@app.post("/correct")
def save_correction(id: int, biopsy: int, mag: int):
    "Model was wrong — use your correction as label"
    db = get_db(DB_PATH)
    image = get_image_by_id(db, id)
    
    if image:
        state.undo_stack.append({
            "idx": state.idx, "action": "correct", "id": id,
            "old_biopsy": image.get("has_biopsy_tool"),
            "old_mag": image.get("has_mag_view"),
        })
        if len(state.undo_stack) > UNDO_STACK_SIZE: state.undo_stack.pop(0)
        
        set_labels(db, id, biopsy, mag)
        state.corrections += 1
    
    state.idx += 1
    state.show_correction = False
    return main_content(db)

@app.post("/skip")
def skip_current():
    db = get_db(DB_PATH)
    state.idx += 1
    state.show_correction = False
    return main_content(db)

@app.post("/undo")
def undo_last():
    db = get_db(DB_PATH)
    
    if state.undo_stack:
        undo = state.undo_stack.pop()
        state.idx = undo["idx"]
        
        if undo["action"] == "confirm":
            state.confirmations -= 1
            db["labels"].update(undo["id"], {"has_biopsy_tool": None, "has_mag_view": None, "labeled_at": None})
        elif undo["action"] == "correct":
            state.corrections -= 1
            db["labels"].update(undo["id"], {"has_biopsy_tool": undo["old_biopsy"], "has_mag_view": undo["old_mag"], "labeled_at": None})
    
    state.show_correction = False
    return main_content(db)

@app.post("/reset")
def reset_queue():
    db = get_db(DB_PATH)
    state.reset()
    state.queue = get_review_queue(db)
    return main_content(db)

@app.get("/stats")
def get_stats_json():
    return {"queue": len(state.queue), "reviewed": state.total, "confirmations": state.confirmations,
            "corrections": state.corrections, "remaining": state.remaining,
            "accuracy": state.confirmations / max(1, state.total)}

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Review UI: http://localhost:{REVIEW_PORT}")
    print("Y=Correct  N=Wrong  S=Skip  U=Undo")
    print("If wrong: 1=Biopsy 2=Mag 3=Neither 4=Both Esc=Cancel")
    serve(port=REVIEW_PORT)
