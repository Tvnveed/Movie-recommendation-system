from __future__ import annotations

import argparse
from pathlib import Path

from movie_recommender.data import ensure_movielens_small
from movie_recommender.recommender import MovieRecommender


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="movie-recommender",
        description="Movie recommendation using MovieLens (collab/content/hybrid).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download-data", help="Download MovieLens latest-small zip and extract it.")
    p_dl.add_argument("--data-dir", type=Path, default=Path("data"))

    p_rec = sub.add_parser("recommend-user", help="Recommend movies for a specific userId.")
    p_rec.add_argument("--data-dir", type=Path, default=Path("data"))
    p_rec.add_argument("--user-id", type=int, required=True)
    p_rec.add_argument("--top-n", type=int, default=10)
    p_rec.add_argument("--method", choices=["hybrid", "collab", "content"], default="hybrid")
    p_rec.add_argument("--min-rating", type=float, default=4.0, help="For content-profile: treat >= as liked.")
    p_rec.add_argument("--alpha", type=float, default=0.7, help="Hybrid weight for collab (0..1).")

    p_sim = sub.add_parser("similar-movies", help="Find movies similar to a title.")
    p_sim.add_argument("--data-dir", type=Path, default=Path("data"))
    p_sim.add_argument("--title", type=str, required=True)
    p_sim.add_argument("--top-n", type=int, default=10)
    p_sim.add_argument("--method", choices=["collab", "content"], default="content")

    return p


def _print_recs(recs: list[tuple[int, float]], r: MovieRecommender) -> None:
    if not recs:
        print("No recommendations found.")
        return
    for rank, (movie_id, score) in enumerate(recs, start=1):
        print(f"{rank:>2}. {r.movie_title(movie_id)} (movieId={movie_id})  score={score:.4f}")


def main() -> int:
    args = _build_parser().parse_args()

    if args.cmd == "download-data":
        path = ensure_movielens_small(args.data_dir)
        print(f"OK: MovieLens extracted to: {path}")
        return 0

    dataset_dir = ensure_movielens_small(args.data_dir)
    r = MovieRecommender.from_movielens(dataset_dir)

    if args.cmd == "recommend-user":
        recs = r.recommend_for_user(
            user_id=args.user_id,
            top_n=args.top_n,
            method=args.method,
            min_like_rating=args.min_rating,
            alpha=args.alpha,
        )
        _print_recs(recs, r)
        return 0

    if args.cmd == "similar-movies":
        recs = r.similar_movies(
            title_query=args.title,
            top_n=args.top_n,
            method=args.method,
        )
        _print_recs(recs, r)
        return 0

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())

