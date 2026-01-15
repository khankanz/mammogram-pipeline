#!/usr/bin/env python3
"""FastHTML labeling server with keyboard-driven UI.

Single keypress labeling - no confirmation needed.
1=Biopsy, 2=MagView, 3=Neither, 4=Both, S=Skip, U=Undo
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent dir to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fasthtml.common import *
from lib.config import LABEL_SERVER_PORT, UNDO_STACK_SIZE, DB_PATH
from lib.db import get_db, get_stats, get_image_by_id, get_unlabeled, set_labels

# Initialize app
app, rt = fast_app(
    hdrs=(
        Style("""
            :root { --bg: #0a0a0a; --fg: #e0e0e0; --accent: #4a9eff; --yes: #22c55e; --skip: #6b7280; }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { background: var(--bg); color: var(--fg); font-family: system-ui, sans-serif; height: 100vh; overflow: hidden; }
            .container { height: 100vh; display: flex; flex-direction: column; padding: 6px; }
            .stats { background: #1a1a1a; padding: 4px 10px; border-radius: 4px; display: flex; gap: 15px; font-size: 11px; flex-shrink: 0; }
            .stat-value { color: var(--accent); font-weight: bold; }
            .image-container { flex: 1; display: flex; justify-content: center; align-items: center; background: #111; border-radius: 4px; margin: 6px 0; min-height: 0; }
            .image-container img { max-height: 100%; max-width: 100%; object-fit: contain; }
            .filename { text-align: center; color: #666; font-size: 10px; font-family: monospace; flex-shrink: 0; }
            .buttons { display: flex; justify-content: center; gap: 8px; flex-shrink: 0; padding: 6px 0; }
            .btn { padding: 6px 12px; font-size: 12px; border: none; border-radius: 4px; cursor: pointer; font-weight: 600; color: white; }
            .btn-biopsy { background: #2563eb; }
            .btn-mag { background: #7c3aed; }
            .btn-neither { background: #374151; }
            .btn-both { background: #0891b2; }
            .btn-skip { background: var(--skip); }
            .btn-undo { background: #dc2626; }
            .kbd { background: #333; padding: 1px 5px; border-radius: 3px; font-family: monospace; margin-right: 3px; font-size: 10px; }
            .empty { text-align: center; padding: 60px; color: #666; }
        """),
    ),
    secret_key=os.urandom(24),
)

# Undo stack
undo_stack = []


def get_next_image(db):
    """Get next unlabeled image."""
    rows = get_unlabeled(db, limit=1)
    return rows[0] if rows else None


def stats_bar(db):
    """Render compact statistics bar."""
    stats = get_stats(db)
    return Div(
        Span(f"Total: ", Span(f"{stats['total']}", cls="stat-value")),
        Span(f"Done: ", Span(f"{stats['fully_labeled']}", cls="stat-value")),
        Span(f"Biopsy: ", Span(f"{stats['biopsy_yes']}", cls="stat-value")),
        Span(f"Mag: ", Span(f"{stats['mag_yes']}", cls="stat-value")),
        Span(f"Left: ", Span(f"{stats['total'] - stats['fully_labeled']}", cls="stat-value")),
        cls="stats", id="stats"
    )


def image_display(image_row):
    """Render image."""
    if not image_row:
        return Div(
            H2("All images labeled!"),
            P("Run training: python scripts/03_train.py"),
            cls="empty"
        )
    return Div(
        Img(src=f"/image/{image_row['id']}", alt=image_row["filename"]),
        cls="image-container"
    )


def label_buttons(image_id):
    """Render labeling buttons."""
    if not image_id:
        return Div()
    return Div(
        Button(Span("1", cls="kbd"), "Biopsy", hx_post=f"/label?id={image_id}&biopsy=1&mag=0", hx_target="#main", cls="btn btn-biopsy"),
        Button(Span("2", cls="kbd"), "MagView", hx_post=f"/label?id={image_id}&biopsy=0&mag=1", hx_target="#main", cls="btn btn-mag"),
        Button(Span("3", cls="kbd"), "Neither", hx_post=f"/label?id={image_id}&biopsy=0&mag=0", hx_target="#main", cls="btn btn-neither"),
        Button(Span("4", cls="kbd"), "Both", hx_post=f"/label?id={image_id}&biopsy=1&mag=1", hx_target="#main", cls="btn btn-both"),
        Button(Span("S", cls="kbd"), "Skip", hx_post=f"/skip?id={image_id}", hx_target="#main", cls="btn btn-skip"),
        Button(Span("U", cls="kbd"), "Undo", hx_post="/undo", hx_target="#main", cls="btn btn-undo"),
        cls="buttons"
    )


def keyboard_script(image_id):
    """JavaScript for keyboard shortcuts."""
    if not image_id:
        return Script("")
    return Script(f"""
        document.addEventListener('keydown', function(e) {{
            if (e.target.tagName === 'INPUT') return;
            const id = {image_id};
            let url = null;
            switch(e.key) {{
                case '1': url = '/label?id=' + id + '&biopsy=1&mag=0'; break;
                case '2': url = '/label?id=' + id + '&biopsy=0&mag=1'; break;
                case '3': url = '/label?id=' + id + '&biopsy=0&mag=0'; break;
                case '4': url = '/label?id=' + id + '&biopsy=1&mag=1'; break;
                case 's': case 'S': url = '/skip?id=' + id; break;
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
    image = get_next_image(db)
    image_id = image["id"] if image else None
    filename = image["filename"].split("/")[-1][:40] if image else ""

    return Div(
        stats_bar(db),
        image_display(image),
        Div(filename, cls="filename") if image else "",
        label_buttons(image_id),
        keyboard_script(image_id),
        id="main"
    )


@rt("/")
def get():
    """Main page."""
    db = get_db(DB_PATH)
    return Html(
        Head(
            Title("DICOM Labeler"),
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


@rt("/label")
def post(id: int, biopsy: int, mag: int):
    """Save label and advance."""
    global undo_stack
    db = get_db(DB_PATH)

    current = get_image_by_id(db, id)
    if current:
        undo_stack.append({
            "id": id,
            "old_biopsy": current["has_biopsy_tool"],
            "old_mag": current["has_mag_view"],
        })
        if len(undo_stack) > UNDO_STACK_SIZE:
            undo_stack.pop(0)

    set_labels(db, id, biopsy, mag)

    return main_content(db)


@rt("/skip")
def post(id: int):
    """Skip image."""
    db = get_db(DB_PATH)
    db["labels"].update(id, {"confidence_biopsy": 999.0})
    return main_content(db)


@rt("/undo")
def post():
    """Undo last label."""
    global undo_stack
    db = get_db(DB_PATH)

    if undo_stack:
        undo = undo_stack.pop()
        db["labels"].update(undo["id"], {
            "has_biopsy_tool": undo["old_biopsy"],
            "has_mag_view": undo["old_mag"],
            "labeled_at": None,
        })

    return main_content(db)


@rt("/stats")
def get():
    """JSON stats endpoint."""
    db = get_db(DB_PATH)
    return get_stats(db)


if __name__ == "__main__":
    print(f"Starting labeling server on http://localhost:{LABEL_SERVER_PORT}")
    print("Keys: 1=Biopsy, 2=MagView, 3=Neither, 4=Both, S=Skip, U=Undo")
    serve(port=LABEL_SERVER_PORT)
