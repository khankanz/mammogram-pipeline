#!usr/bin/env python3
"""FastHTML labeling server with keyboard-driven UI

Single keypress labeling - no confirmation needed.
1=Biopsy, 2=MagView, 3=Neither, 4=Both, S=Skip, U=Undo

Refactored to follow FastHTML idioms:
- Route style (@app.get/@app.post with descriptive names)
- Let FastHTML handle HTML wrapper (return Title, content)
- Single keyboard listener defined in hdrs (no listener stacking!)
- Uses data-* attributes for dynamic values
"""

import os, sys
from pathlib import Path

# This adds the project root to Python's import path so you can do; from lib.config import ...
# Without it Python wouldn't find the lib package since we're running from scripts/ 
# The alternatively would be to install package via pip install -e .
sys.path.insert(0, str(Path(__file__).parent.parent)) 

from fasthtml.common import *
from lib.config import LABEL_SERVER_PORT, UNDO_STACK_SIZE, DB_PATH
from lib.db import get_db, get_stats, get_image_by_id, get_unlabeled, set_labels

# =============================================================================
# STYLING
# =============================================================================

_styles = Style("""
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
""")

# =============================================================================
# KEYBOARD HANDLER - DEFINED ONCE, READS DATA ATTRIBUTES
# =============================================================================

_keyboard_handler = Script("""
document.addEventListener('keydown', function(e) {
    // Don't intercept when typing in inputs
    if (e.target.tagName === 'INPUT') return;

    // Read current image ID from data attribute on #main
    const mainEl = document.getElementById('main-inner');
    const imageId = mainEl?.dataset?.imageId; 
    
    // oh how Iv'e missed the ? notation in JavaScript. This is Optional Chaining. mainEl might be null which means if we try to get dataset attr, it goes boom. In JS we can chain and the moment it hits null or undefined, it stops and returns undefined. Python doesn't have an equivalent. The same thing would be; IF mainEl and dataset is not None then grab it or use getattr with default None values OR a try/catch block then there's also dict.get() which I've unfortunately used too much of, nothing wrong with it just reads ugly.

    // Build route map - undo doesn't need imageId
    let url = null;
    const key = e.key.toLowerCase();

    if (key == 'u') {
        url = '/undo';
    } else if (imageId && imageId !== 'None' && imageId !== '') {
        // These routes require an image ID
        const routes = {
            '1': `/label?id=${imageId}&biopsy=1&mag=0`,
            '2': `/label?id=${imageId}&biopsy=0&mag=1`,
            '3': `/label?id=${imageId}&biopsy=0&mag=0`,
            '4': `/label?id=${imageId}&biopsy=1&mag=1`,
            's': `/skip?id=${imageId}`
        };
        url = routes[key];
    }

    if (url){
        e.preventDefault(); // Ah this is important; forms be doing some weird shit by default. Without this Browser might try to submit a form, some keys i.e. spacebar have default behaviour, we'd get double-actions. This says yo browswer, don't do SOP pls.
        htmx.ajax('POST', url, {target: '#main', swap: 'innerHTML'});
    }   
});
""")

# =============================================================================
# APP INIT
# =============================================================================

app, rt = fast_app(
    hdrs=(_styles, _keyboard_handler),
    secret_key=os.urandom(24), # Why? For session cookies signing; FastHTML via Starlette can store data in the user's browser via signed cookies
    # NOW this approach will generate a NEW key every time the server restarts
)

# Undo stack (module-level state for single-user tool)
undo_stack = []

# =============================================================================
# COMPONENTS
# =============================================================================

# This method inherits get_unlabeled from lib.db.py's logic i.e. 10% random (exploration), 45% uncertain (where our labels have max info value) and 45% condifdent (catches systemic errors)
def get_next_image(db): 
    """Get next unlabeled image via active learning selection"""
    rows = get_unlabeled(db, limit=1)
    return rows[0] if rows else None

def stats_bar(db):
    """Compact statistics bar"""
    s = get_stats(db)
    return Div(
        Span("Total: ", Span(s['total'], cls="stat-value")),
        Span('Done: ', Span(s['fully_labeled'], cls="stat-value")),
        Span('Biopsy: ', Span(s['biopsy_yes'], cls="stat-value")),
        Span('Mag: ', Span(s['mag_yes'], cls="stat-value")),
        Span('Left: ', Span(s['total'] - s['fully_labeled'], cls="stat-value")),
        cls="stats", id="stats"
    )

def image_display(image_row):
    """Render the image or completion message."""
    if not image_row:
        return Div(
            H2("All images labeled!"),
            P("Run training: python scripts/03_train.py"),
            cls="empty"
        )
    return Div(
        Img(src=f"/image/{image_row['id']}", alt=image_row['filename']),
        cls="image-container"
    )

def label_buttons(image_id):
    """Render labeling buttons with HTMX triggers"""
    if not image_id: return Div()
    return Div(
        Button(Span("1", cls="kbd"), "Biopsy",
                hx_post=f"/label?id={image_id}&biopsy=1&mag=0", hx_target="#main", cls="btn btn-biopsy"),
        Button(Span("2", cls="kbd"), "MagView",
                hx_post=f"/label?id={image_id}&biopsy=0&mag=1", hx_target="#main", cls="btn btn-mag"),
        Button(Span("3", cls="kbd"), "Neither",
                hx_post=f"/label?id={image_id}&biopsy=0&mag=0", hx_target="#main", cls="btn btn-neither"),
        Button(Span("4", cls="kbd"), "Both",
                hx_post=f"/label?id={image_id}&biopsy=1&mag=1", hx_target="#main", cls="btn btn-both"),
        Button(Span("S", cls="kbd"), "Skip",
                hx_post=f"/skip?id={image_id}", hx_target="#main", cls="btn btn-skip"),
        Button(Span("U", cls="kbd"), "Undo",
                hx_post="/undo", hx_target="#main", cls="btn btn-undo"),
        cls="buttons"
    )

def main_content(db, override_image=None):
    """Main UI content - stats, image, buttons.
    
    The data_image_id attribute is read by keyboard handler JS.
    This is how we pass dynamic data to a single global listener.

    Now accepts an override_image param.
    """
    image = override_image if override_image else get_next_image(db)
    image_id = image["id"] if image else None
    filename = image["filename"].split("/")[-1][:40] if image else ""

    data_id = str(image_id) if image_id else "" # convert None to empty string for JS compatibility; without this image_id is None, Python renders "None" the string, then JS would try to POST /label?id=None&biopsy= which is like boom

    return Div(
        stats_bar(db),
        image_display(image),
        Div(filename, cls="filename") if image else "",
        label_buttons(image_id),
        id="main-inner",
        data_image_id=image_id, # JS reads this via mainEl.dataset.imageId
    )

# =============================================================================
# ROUTES - Style B: @app.get/@app.post with descriptive function names
# =============================================================================

@app.get("/")
def homepage():
    """Main page - FastHTML auto-wraps in HTML with headers."""
    db = get_db(DB_PATH)
    return Title("FastHTML Labeler"), Div(Div(main_content(db), id="main"), cls="container")

# This method is fetching the image from the database path so we can populate the labeling interface
@app.get("/image/{image_id}")
def serve_thumbnail(image_id:int):
    """Serve thumbnail image from database path"""
    db = get_db(DB_PATH)
    image = get_image_by_id(db, image_id)
    if not image: return Response("Not found", status_code=404)
    thumb_path = Path(image["thumbnail_path"])
    if not thumb_path.exists(): return Response("Thumbnail not found", status_code=404)
    return FileResponse(thumb_path, media_type="image/png")

@app.post("/label")
def save_label(id:int, biopsy:int, mag:int):
    """Save label and advance to next image."""
    global undo_stack
    db = get_db(DB_PATH)

    # Save current state for undo
    current = get_image_by_id(db, id)
    if current:
        undo_stack.append({
            "id": id, "old_biopsy": current["has_biopsy_tool"], "old_mag": current["has_mag_view"],})
        if len(undo_stack) > UNDO_STACK_SIZE: undo_stack.pop(0)
    
    # Save label (also assigns trai/val split on first label)
    set_labels(db, id, biopsy, mag)
    return main_content(db)

# Why? This is a sentient value hack; 
@app.post("/skip")
def skip_image(id:int):
    """Skip image - just move on without labeling, but allow undo"""
    global undo_stack
    db = get_db(DB_PATH)
    
    # Save to undo stack so we can return to this image
    undo_stack.append({"id": id, "action": "skip"}) # Mark as skip, not a label change 
    if len(undo_stack) > UNDO_STACK_SIZE: undo_stack.pop(0)
    return main_content(db)

@app.post("/undo")
def undo_last():
    """Undo last label"""
    global undo_stack
    db = get_db(DB_PATH)

    if undo_stack:
        undo = undo_stack.pop()
        if undo.get("action") == "skip":
            # For skip, just return that image
            image = get_image_by_id(db, undo["id"])
            return main_content(db, override_image=image)
        else:
            db["labels"].update(undo["id"], {
                "has_biopsy_tool": undo["old_biopsy"],
                "has_mag_view": undo["old_mag"],
                "labeled_at": None
            })
            # Return the image we just undid!
            undone_image = get_image_by_id(db, undo["id"])
            return main_content(db, override_image=undone_image)
            
    return main_content(db)

@app.get("/stats")
def get_stats_json():
    """JSON stats endpoint for external monitoring"""
    db = get_db(DB_PATH)
    return get_stats(db)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # I should be using logging instead of print statements
    print(f"Starting labeling server on http://localhost:{LABEL_SERVER_PORT}")
    print("Keys: 1=Biopsy, 2=MagView, 3=Neither, 4=Both, S=Skip, U=Undo")
    serve(port=LABEL_SERVER_PORT)