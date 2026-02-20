from __future__ import annotations

import csv
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_SMALL_DIRNAME = "ml-latest-small"


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title: str
    genres: str


def ensure_movielens_small(data_dir: Path) -> Path:
    """
    Ensures MovieLens (latest-small) is present under data_dir and returns the dataset folder.
    Layout after extraction:
      data_dir/ml-latest-small/movies.csv
      data_dir/ml-latest-small/ratings.csv
    """
    data_dir = Path(data_dir)
    dataset_dir = data_dir / MOVIELENS_SMALL_DIRNAME
    movies_csv = dataset_dir / "movies.csv"
    ratings_csv = dataset_dir / "ratings.csv"
    if movies_csv.exists() and ratings_csv.exists():
        return dataset_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-latest-small.zip"

    print(f"Downloading MovieLens to {zip_path} ...", file=sys.stderr)
    with urllib.request.urlopen(MOVIELENS_SMALL_URL) as resp:
        content = resp.read()
    zip_path.write_bytes(content)

    print(f"Extracting to {data_dir} ...", file=sys.stderr)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    if not (movies_csv.exists() and ratings_csv.exists()):
        raise RuntimeError(f"Extraction did not produce expected files under {dataset_dir}")

    return dataset_dir


def load_movies_csv(dataset_dir: Path) -> dict[int, Movie]:
    path = Path(dataset_dir) / "movies.csv"
    movies: dict[int, Movie] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = int(row["movieId"])
            movies[mid] = Movie(movie_id=mid, title=row["title"], genres=row["genres"])
    return movies


def load_ratings_csv(dataset_dir: Path) -> list[tuple[int, int, float]]:
    """
    Returns list of (userId, movieId, rating).
    """
    path = Path(dataset_dir) / "ratings.csv"
    ratings: list[tuple[int, int, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratings.append((int(row["userId"]), int(row["movieId"]), float(row["rating"])))
    return ratings


def normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


def movie_text(m: Movie) -> str:
    genres = m.genres.replace("|", " ")
    return normalize_text(f"{m.title} {genres}")

