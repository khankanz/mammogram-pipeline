# FastHTML Labeling Server: Keyboard-Driven UI

A single-page labeling interface built with FastHTML. One keypress per label, no confirmation dialogs, undo support. Designed for speed when you need to label hundreds of images.

## The Design Philosophy

Traditional labeling UIs are click-heavy: click image, click label, click confirm, click next. That's four actions per image. At 500 images, you've clicked 2000 times.

This interface: one keypress. `1` for biopsy, `2` for mag view, `3` for neither, `4` for both. No confirmation. Undo with `U` if you mess up.

---

## Project Setup: The Import Path Trick

```python
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
```

This adds the project root to Python's import path so you can do `from lib.config import ...`. Without it, Python wouldn't find the `lib` package since we're running from `scripts/`. The alternative would be installing the package via `pip install -e .`, but this is simpler for a single-user tool.

---

## Styling: Dark Theme, Minimal Chrome

```python
_styles = Style("""
    :root { --bg: #0a0a0a; --fg: #e0e0e0; --accent: #4a9eff; --yes: #22c55e; --skip: #6b7280; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--fg); font-family: system-ui, sans-serif; 
           height: 100vh; overflow: hidden; }
    .container { height: 100vh; display: flex; flex-direction: column; padding: 6px; }
    /* ... */
""")
```

CSS custom properties (`--bg`, `--accent`) make theming trivial. The layout is a flexbox column: stats bar at top, image in the middle (flex-grow to fill space), buttons at bottom. `overflow: hidden` on body prevents scroll bounce.

---

## The Keyboard Handler: One Listener, Data Attributes

Here's the trick that makes keyboard handling clean. Instead of attaching new listeners on every HTMX swap (which leads to listener stacking and double-fires), we define ONE global listener that reads the current image ID from a data attribute:

```javascript
document.addEventListener('keydown', function(e) {
    // Don't intercept when typing in inputs
    if (e.target.tagName === 'INPUT') return;

    // Read current image ID from data attribute on #main-inner
    const mainEl = document.getElementById('main-inner');
    const imageId = mainEl?.dataset?.imageId;
```

The `?.` is JavaScript's optional chaining—if `mainEl` is null, it stops and returns undefined instead of throwing. Python doesn't have an equivalent; you'd need `getattr(obj, 'attr', None)` or a try/catch.

```javascript
    // Build route map - undo doesn't need imageId
    let url = null;
    const key = e.key.toLowerCase();

    if (key == 'u') {
        url = '/undo';
    } else if (imageId && imageId !== 'None' && imageId !== '') {
        const routes = {
            '1': `/label?id=${imageId}&biopsy=1&mag=0`,
            '2': `/label?id=${imageId}&biopsy=0&mag=1`,
            '3': `/label?id=${imageId}&biopsy=0&mag=0`,
            '4': `/label?id=${imageId}&biopsy=1&mag=1`,
            's': `/skip?id=${imageId}`
        };
        url = routes[key];
    }

    if (url) {
        e.preventDefault();
        htmx.ajax('POST', url, {target: '#main', swap: 'innerHTML'});
    }
});
```

The `e.preventDefault()` is important—without it, the browser might try to submit a form, spacebar would scroll, etc. We're saying "browser, don't do your standard thing, I've got this."

---

## App Initialization

```python
app, rt = fast_app(
    hdrs=(_styles, _keyboard_handler),
    secret_key=os.urandom(24),
)
```

`hdrs` injects our styles and keyboard handler into every page's `<head>`. The `secret_key` is for session cookie signing—FastHTML (via Starlette) can store data in the user's browser via signed cookies. Note: `os.urandom(24)` generates a NEW key every restart, so sessions don't persist across server restarts. Fine for a single-user tool.

```python
# Undo stack (module-level state for single-user tool)
undo_stack = []
```

Module-level state works here because it's a single-user tool. For multi-user, you'd use session storage or a database table.

---

## Components: Building Blocks

**Stats Bar:**
```python
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
```

**Image Display:**
```python
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
```

**Label Buttons:**
```python
def label_buttons(image_id):
    """Render labeling buttons with HTMX triggers"""
    if not image_id: return Div()
    return Div(
        Button(Span("1", cls="kbd"), "Biopsy",
               hx_post=f"/label?id={image_id}&biopsy=1&mag=0", 
               hx_target="#main", cls="btn btn-biopsy"),
        # ... more buttons
        cls="buttons"
    )
```

Each button has `hx_post` and `hx_target`—HTMX will POST to that URL and swap the response into `#main`. The keyboard handler does the same thing via `htmx.ajax()`.

---

## Main Content: The Data Attribute Bridge

This is where the keyboard handler gets its data:

```python
def main_content(db, override_image=None):
    """Main UI content - stats, image, buttons."""
    image = override_image if override_image else get_next_image(db)
    image_id = image["id"] if image else None
    filename = image["filename"].split("/")[-1][:40] if image else ""

    # Convert None to empty string for JS compatibility
    data_id = str(image_id) if image_id else ""

    return Div(
        stats_bar(db),
        image_display(image),
        Div(filename, cls="filename") if image else "",
        label_buttons(image_id),
        id="main-inner",
        data_image_id=image_id,  # JS reads this via mainEl.dataset.imageId
    )
```

The `data_image_id` attribute is the bridge between Python and JavaScript. When HTMX swaps in new content, the new `data-image-id` value is already there for the keyboard handler to read.

The `override_image` parameter is for undo—we want to show the image we just undid, not fetch the next one from active learning.

Why convert `None` to empty string? Without it, `image_id` is Python's `None`, which renders as the string `"None"`. Then JS would try to POST `/label?id=None&biopsy=...` which is bad.

---

## Routes: Style B

FastHTML supports two routing styles. This codebase uses Style B—`@app.get`/`@app.post` with descriptive function names:

```python
@app.get("/")
def homepage():
    """Main page - FastHTML auto-wraps in HTML with headers."""
    db = get_db(DB_PATH)
    return Title("FastHTML Labeler"), Div(Div(main_content(db), id="main"), cls="container")
```

FastHTML sees the `Title` component and automatically wraps everything in a proper HTML document with `<head>` containing our styles and scripts.

**Image Serving:**
```python
@app.get("/image/{image_id}")
def serve_thumbnail(image_id: int):
    """Serve thumbnail image from database path"""
    db = get_db(DB_PATH)
    image = get_image_by_id(db, image_id)
    if not image: return Response("Not found", status_code=404)
    thumb_path = Path(image["thumbnail_path"])
    if not thumb_path.exists(): return Response("Thumbnail not found", status_code=404)
    return FileResponse(thumb_path, media_type="image/png")
```

The database stores the path; this route serves the actual file.

---

## Labeling Flow

**Save Label:**
```python
@app.post("/label")
def save_label(id: int, biopsy: int, mag: int):
    """Save label and advance to next image."""
    global undo_stack
    db = get_db(DB_PATH)

    # Save current state for undo
    current = get_image_by_id(db, id)
    if current:
        undo_stack.append({
            "id": id, 
            "old_biopsy": current["has_biopsy_tool"], 
            "old_mag": current["has_mag_view"],
        })
        if len(undo_stack) > UNDO_STACK_SIZE: undo_stack.pop(0)
    
    # Save label (also assigns train/val split on first label)
    set_labels(db, id, biopsy, mag)
    return main_content(db)
```

The undo stack stores the *previous* state before we overwrite it. `set_labels()` handles the train/val split assignment via `_ensure_split()`.

**Skip:**
```python
@app.post("/skip")
def skip_image(id: int):
    """Skip image - just move on without labeling, but allow undo"""
    global undo_stack
    db = get_db(DB_PATH)
    
    # Save to undo stack so we can return to this image
    undo_stack.append({"id": id, "action": "skip"})
    if len(undo_stack) > UNDO_STACK_SIZE: undo_stack.pop(0)
    return main_content(db)
```

Skip is a "sentinel value hack"—we mark it differently in the undo stack so undo knows to just return to that image rather than restore old labels.

**Undo:**
```python
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
            # Restore previous state
            db["labels"].update(undo["id"], {
                "has_biopsy_tool": undo["old_biopsy"],
                "has_mag_view": undo["old_mag"],
                "labeled_at": None
            })
            # Return the image we just undid
            undone_image = get_image_by_id(db, undo["id"])
            return main_content(db, override_image=undone_image)
            
    return main_content(db)
```

The `override_image` parameter is crucial—without it, `main_content()` would call `get_next_image()` which uses active learning to pick the next image. We want to show the specific image we just undid.

---

## Active Learning Integration

```python
def get_next_image(db): 
    """Get next unlabeled image via active learning selection"""
    rows = get_unlabeled(db, limit=1)
    return rows[0] if rows else None
```

This inherits the sampling logic from `lib.db.get_unlabeled()`:
- 10% random (exploration)
- 45% uncertain (confidence ≈ 0.5, maximum information value)
- 45% confident (confidence ≈ 0 or 1, catches systematic errors)

You don't see random images after the first training run—you see the ones where your labels will help the model most.

---

## Running It

```bash
python scripts/02_label_server.py
```

Output:
```
Starting labeling server on http://localhost:5001
Keys: 1=Biopsy, 2=MagView, 3=Neither, 4=Both, S=Skip, U=Undo
```

Open `http://localhost:5001` in your browser. Start pressing keys.

---

## Key Takeaways

1. **One global keyboard listener** that reads data attributes—no listener stacking on HTMX swaps
2. **Data attributes as the Python→JS bridge**—`data_image_id` carries state across swaps
3. **Undo stack with sentinel values**—skip vs label are handled differently
4. **`override_image` parameter**—lets undo show the specific image, not the next active learning pick
5. **FastHTML auto-wrapping**—return `Title` + content, get a full HTML document with headers

That's the labeling server. One keypress per image, active learning selection, undo support. Fast enough to label hundreds of images without wanting to throw your keyboard.