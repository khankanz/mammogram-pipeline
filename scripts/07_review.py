#!/usr/bin/env python3
"""Review interface for model predictions.

Shows random samples from inference results, lets user confirm or correct.
Corrections are added to training data for active learning.

Keys: Y=Correct, N=Wrong (then select correct label), S=Skip, U=Undo
"""

import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from fasthtml.common import *
from lib.config import LABEL_SERVER_PORT, UNDO_STACK_SIZE, DB_PATH, THUMBNAIL_DIR
from lib.db import get_db, get_stats, get_image_by_id, set_labels

REVIEW_PORT = 5002

app, rt = fast_app(
    hdrs=(
        Style("""
            :root { --bg: #0a0a0a; --fg: #e0e0e0; --accent: #4a9eff; --correct: #22c55e; --wrong: #ef4444; }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { background: var(--bg); color: var(--fg); font-family: system-ui, sans-serif; height: 100vh; overflow: hidden; }
            .container { height: 100vh; display: flex; flex-direction: column; padding: 6px; }
            .stats { background: #1a1a1a; padding: 4px 10px; border-radius: 4px; display: flex; gap: 15px; font-size: 11px; flex-shrink: 0; }
            .stat-value { color: var(--accent); font-weight: bold; }
            .image-container { flex: 1; display: flex; justify-content: center; align-items: center; background: #111; border-radius: 4px; margin: 6px 0; min-height: 0; }
            .image-container img { max-height: 100%; max-width: 100%; object-fit: contain; }
            .prediction { text-align: center; padding: 8px; font-size: 14px; flex-shrink: 0; }
            .pred-label { display: inline-block; padding: 4px 12px; border-radius: 4px; margin: 0 4px; }
            .pred-yes { background: #166534; }
            .pred-no { background: #374151; }
            .pred-prob { color: #888; font-size: 11px; }
            .buttons { display: flex; justify-content: center; gap: 8px; flex-shrink: 0; padding: 6px 0; }
            .btn { padding: 8px 16px; font-size: 13px; border: none; border-radius: 4px; cursor: pointer; font-weight: 600; color: white; }
            .btn-correct { background: var(--correct); }
            .btn-wrong { background: var(--wrong); }
            .btn-skip { background: #6b7280; }
            .btn-undo { background: #7c3aed; }
            .correction-panel { background: #1a1a1a; padding: 10px; border-radius: 4px; margin: 6px 0; }
            .correction-title { font-size: 12px; color: #888; margin-bottom: 8px; }
            .kbd { background: #333; padding: 1px 5px; border-radius: 3px; font-family: monospace; margin-right: 3px; font-size: 10px; }
            .empty { text-align: center; padding: 60px; color: #666; }
            .filename { text-align: center; color: #666; font-size: 10px; font-family: monospace; flex-shrink: 0; }
        """),
    ),
    secret_key=os.urandom(24),
)

# State
review_state = {
    "queue": [],  # List of image IDs to review
    "current_idx": 0,
    "undo_stack": [],
    "corrections": 0,
    "confirmations": 0,
    "show_correction": False,
}


def get_review_queue(db, limit=100):
    """Get images with predictions for review, randomly sampled."""
    import random

    # Get images that have predictions but we want to verify
    rows = list(db["labels"].rows_where(
        "confidence_biopsy IS NOT NULL AND confidence_mag IS NOT NULL "
        "AND (split IS NULL OR split IN ('train', 'val'))",
        order_by="RANDOM()",
        limit=limit
    ))
    return [r["id"] for r in rows]


def get_current_image(db):
    """Get current image to review."""
    if review_state["current_idx"] >= len(review_state["queue"]):
        return None
    image_id = review_state["queue"][review_state["current_idx"]]
    return get_image_by_id(db, image_id)


def stats_bar(db):
    """Render statistics bar."""
    total_reviewed = review_state["corrections"] + review_state["confirmations"]
    remaining = len(review_state["queue"]) - review_state["current_idx"]

    return Div(
        Span(f"Queue: ", Span(f"{len(review_state['queue'])}", cls="stat-value")),
        Span(f"Reviewed: ", Span(f"{total_reviewed}", cls="stat-value")),
        Span(f"Correct: ", Span(f"{review_state['confirmations']}", cls="stat-value")),
        Span(f"Corrected: ", Span(f"{review_state['corrections']}", cls="stat-value")),
        Span(f"Remaining: ", Span(f"{remaining}", cls="stat-value")),
        cls="stats", id="stats"
    )


def prediction_display(image_row):
    """Show model predictions."""
    if not image_row:
        return Div()

    biopsy_prob = image_row.get("confidence_biopsy", 0) or 0
    mag_prob = image_row.get("confidence_mag", 0) or 0

    biopsy_pred = biopsy_prob >= 0.5
    mag_pred = mag_prob >= 0.5

    return Div(
        Span("Model prediction: "),
        Span(
            f"Biopsy: {'YES' if biopsy_pred else 'NO'}",
            cls=f"pred-label {'pred-yes' if biopsy_pred else 'pred-no'}"
        ),
        Span(f"({biopsy_prob:.1%})", cls="pred-prob"),
        Span(" "),
        Span(
            f"MagView: {'YES' if mag_pred else 'NO'}",
            cls=f"pred-label {'pred-yes' if mag_pred else 'pred-no'}"
        ),
        Span(f"({mag_prob:.1%})", cls="pred-prob"),
        cls="prediction"
    )


def correction_panel(image_id):
    """Panel for selecting correct label when prediction is wrong."""
    return Div(
        Div("Select correct label:", cls="correction-title"),
        Div(
            Button(Span("1", cls="kbd"), "Biopsy Only",
                   hx_post=f"/correct?id={image_id}&biopsy=1&mag=0", hx_target="#main",
                   cls="btn", style="background: #2563eb;"),
            Button(Span("2", cls="kbd"), "MagView Only",
                   hx_post=f"/correct?id={image_id}&biopsy=0&mag=1", hx_target="#main",
                   cls="btn", style="background: #7c3aed;"),
            Button(Span("3", cls="kbd"), "Neither",
                   hx_post=f"/correct?id={image_id}&biopsy=0&mag=0", hx_target="#main",
                   cls="btn", style="background: #374151;"),
            Button(Span("4", cls="kbd"), "Both",
                   hx_post=f"/correct?id={image_id}&biopsy=1&mag=1", hx_target="#main",
                   cls="btn", style="background: #0891b2;"),
            Button("Cancel", hx_post="/cancel", hx_target="#main", cls="btn btn-skip"),
            cls="buttons"
        ),
        cls="correction-panel"
    )


def main_buttons(image_id):
    """Main review buttons."""
    if not image_id:
        return Div()

    if review_state["show_correction"]:
        return correction_panel(image_id)

    return Div(
        Button(Span("Y", cls="kbd"), "Correct", hx_post=f"/confirm?id={image_id}", hx_target="#main", cls="btn btn-correct"),
        Button(Span("N", cls="kbd"), "Wrong", hx_post="/show-correction", hx_target="#main", cls="btn btn-wrong"),
        Button(Span("S", cls="kbd"), "Skip", hx_post="/skip", hx_target="#main", cls="btn btn-skip"),
        Button(Span("U", cls="kbd"), "Undo", hx_post="/undo", hx_target="#main", cls="btn btn-undo"),
        cls="buttons"
    )


def keyboard_script(image_id):
    """JavaScript for keyboard shortcuts."""
    if not image_id:
        return Script("")

    if review_state["show_correction"]:
        return Script(f"""
            document.addEventListener('keydown', function(e) {{
                const id = {image_id};
                let url = null;
                switch(e.key) {{
                    case '1': url = '/correct?id=' + id + '&biopsy=1&mag=0'; break;
                    case '2': url = '/correct?id=' + id + '&biopsy=0&mag=1'; break;
                    case '3': url = '/correct?id=' + id + '&biopsy=0&mag=0'; break;
                    case '4': url = '/correct?id=' + id + '&biopsy=1&mag=1'; break;
                    case 'Escape': url = '/cancel'; break;
                }}
                if (url) {{
                    e.preventDefault();
                    htmx.ajax('POST', url, {{target: '#main', swap: 'innerHTML'}});
                }}
            }});
        """)

    return Script(f"""
        document.addEventListener('keydown', function(e) {{
            const id = {image_id};
            let url = null;
            switch(e.key) {{
                case 'y': case 'Y': url = '/confirm?id=' + id; break;
                case 'n': case 'N': url = '/show-correction'; break;
                case 's': case 'S': url = '/skip'; break;
                case 'u': case 'U': url = '/undo'; break;
            }}
            if (url) {{
                e.preventDefault();
                htmx.ajax('POST', url, {{target: '#main', swap: 'innerHTML'}});
            }}
        }});
    """)


def main_content(db):
    """Render main interface."""
    # Initialize queue if empty
    if not review_state["queue"]:
        review_state["queue"] = get_review_queue(db)
        review_state["current_idx"] = 0

    image = get_current_image(db)
    image_id = image["id"] if image else None

    if not image:
        return Div(
            stats_bar(db),
            Div(
                H2("Review complete!"),
                P(f"Confirmed: {review_state['confirmations']}, Corrected: {review_state['corrections']}"),
                P("Run training to update model: python scripts/03_train.py"),
                cls="empty"
            ),
            id="main"
        )

    filename = image["filename"].split("/")[-1][:50] if image else ""

    return Div(
        stats_bar(db),
        Div(Img(src=f"/image/{image_id}", alt=filename), cls="image-container"),
        prediction_display(image),
        Div(filename, cls="filename"),
        main_buttons(image_id),
        keyboard_script(image_id),
        id="main"
    )


@rt("/")
def get():
    """Main page."""
    db = get_db(DB_PATH)
    return Html(
        Head(
            Title("Review Predictions"),
            Script(src="https://unpkg.com/htmx.org@1.9.10"),
        ),
        Body(Div(main_content(db), cls="container"))
    )


@rt("/image/{image_id}")
def get(image_id: int):
    """Serve thumbnail image."""
    db = get_db(DB_PATH)
    image = get_image_by_id(db, image_id)
    if not image:
        return Response("Not found", status_code=404)
    thumb_path = Path(image["thumbnail_path"])
    if not thumb_path.exists():
        return Response("Thumbnail not found", status_code=404)
    return FileResponse(thumb_path, media_type="image/png")


@rt("/confirm")
def post(id: int):
    """Confirm prediction is correct."""
    db = get_db(DB_PATH)

    image = get_image_by_id(db, id)
    if image:
        # Save undo state
        review_state["undo_stack"].append({
            "idx": review_state["current_idx"],
            "action": "confirm",
            "id": id,
        })

        # Use prediction as label (confirm it)
        biopsy = 1 if (image.get("confidence_biopsy", 0) or 0) >= 0.5 else 0
        mag = 1 if (image.get("confidence_mag", 0) or 0) >= 0.5 else 0

        set_labels(db, id, biopsy, mag)

        review_state["confirmations"] += 1

    review_state["current_idx"] += 1
    review_state["show_correction"] = False

    return main_content(db)


@rt("/show-correction")
def post():
    """Show correction panel."""
    db = get_db(DB_PATH)
    review_state["show_correction"] = True
    return main_content(db)


@rt("/cancel")
def post():
    """Cancel correction."""
    db = get_db(DB_PATH)
    review_state["show_correction"] = False
    return main_content(db)


@rt("/correct")
def post(id: int, biopsy: int, mag: int):
    """Save corrected label."""
    db = get_db(DB_PATH)

    # Save undo state
    image = get_image_by_id(db, id)
    if image:
        review_state["undo_stack"].append({
            "idx": review_state["current_idx"],
            "action": "correct",
            "id": id,
            "old_biopsy": image.get("has_biopsy_tool"),
            "old_mag": image.get("has_mag_view"),
        })

    set_labels(db, id, biopsy, mag)

    review_state["corrections"] += 1
    review_state["current_idx"] += 1
    review_state["show_correction"] = False

    return main_content(db)


@rt("/skip")
def post():
    """Skip current image."""
    db = get_db(DB_PATH)
    review_state["current_idx"] += 1
    review_state["show_correction"] = False
    return main_content(db)


@rt("/undo")
def post():
    """Undo last action."""
    db = get_db(DB_PATH)

    if review_state["undo_stack"]:
        undo = review_state["undo_stack"].pop()
        review_state["current_idx"] = undo["idx"]

        if undo["action"] == "confirm":
            review_state["confirmations"] -= 1
            # Clear the label
            db["labels"].update(undo["id"], {
                "has_biopsy_tool": None,
                "has_mag_view": None,
                "labeled_at": None
            })
        elif undo["action"] == "correct":
            review_state["corrections"] -= 1
            db["labels"].update(undo["id"], {
                "has_biopsy_tool": undo["old_biopsy"],
                "has_mag_view": undo["old_mag"],
                "labeled_at": None
            })

    review_state["show_correction"] = False
    return main_content(db)


@rt("/reset")
def post():
    """Reset review queue."""
    db = get_db(DB_PATH)
    review_state["queue"] = get_review_queue(db)
    review_state["current_idx"] = 0
    review_state["corrections"] = 0
    review_state["confirmations"] = 0
    review_state["show_correction"] = False
    return main_content(db)


if __name__ == "__main__":
    print(f"Starting review server on http://localhost:{REVIEW_PORT}")
    print("Keys: Y=Correct, N=Wrong, S=Skip, U=Undo")
    print("If wrong: 1=Biopsy, 2=MagView, 3=Neither, 4=Both")
    serve(port=REVIEW_PORT)
