# Movie Recommendation (ML, Python)

This is a small **machine-learning-style** movie recommender using the **MovieLens** dataset.

- **Collaborative filtering (item–item)**: cosine similarity of movie rating vectors.
- **Content-based**: TF‑IDF over movie title + genres.
- **Hybrid**: combines both signals.

No external dependencies (standard library only).

## Quick start (Windows PowerShell)

From this folder:

```powershell
python --version
python .\main.py download-data
python .\main.py recommend-user --user-id 1 --top-n 10 --method hybrid
python .\main.py similar-movies --title "Toy Story" --top-n 10 --method content
```

The dataset is downloaded to `.\data\ml-latest-small\` the first time you run `download-data`.

## Frontend (HTML UI)

Start the local server:

```powershell
python .\server.py
```

Then open:

- `http://127.0.0.1:8000/`

## Commands

### Download MovieLens (required once)

```powershell
python .\main.py download-data --data-dir .\data
```

### Recommend movies for a user

```powershell
python .\main.py recommend-user --user-id 1 --top-n 10 --method hybrid --data-dir .\data
```

Methods:
- `collab`: item–item collaborative filtering only
- `content`: TF‑IDF content-based only
- `hybrid`: combines both (default)

### Find similar movies by title

```powershell
python .\main.py similar-movies --title "Matrix" --top-n 10 --method content --data-dir .\data
```

Tip: title matching is fuzzy (substring), and if multiple movies match, you’ll be prompted to choose.

## Notes

- MovieLens “latest-small” is great for demos; for bigger datasets, you’d typically add caching and more efficient similarity search.

