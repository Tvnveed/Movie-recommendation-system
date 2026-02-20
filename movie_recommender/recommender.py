from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from movie_recommender.data import Movie, load_movies_csv, load_ratings_csv, movie_text, normalize_text
from movie_recommender.tfidf import TfidfModel, build_profile, cosine_sim, fit_tfidf


def _l2_norm_sparse(vec: dict[int, float]) -> float:
    return math.sqrt(sum(v * v for v in vec.values()))


def _dot_sparse(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def cosine_sim_sparse(a: dict[int, float], a_norm: float, b: dict[int, float], b_norm: float) -> float:
    if a_norm <= 0.0 or b_norm <= 0.0:
        return 0.0
    return _dot_sparse(a, b) / (a_norm * b_norm)


@dataclass(frozen=True)
class _Indexes:
    movies: dict[int, Movie]
    # ratings:
    # movie_ratings[movieId][userId] = rating
    movie_ratings: dict[int, dict[int, float]]
    movie_avg_rating: dict[int, float]
    movie_rating_count: dict[int, int]
    movie_norms: dict[int, float]
    # user_ratings[userId][movieId] = rating
    user_ratings: dict[int, dict[int, float]]
    # inverted index: user_movies[userId] = set(movieId)
    user_movies: dict[int, set[int]]
    # content model
    tfidf: TfidfModel


class MovieRecommender:
    def __init__(self, idx: _Indexes):
        self._idx = idx

    @classmethod
    def from_movielens(cls, dataset_dir: Path) -> "MovieRecommender":
        dataset_dir = Path(dataset_dir)
        movies = load_movies_csv(dataset_dir)
        ratings = load_ratings_csv(dataset_dir)

        movie_ratings: dict[int, dict[int, float]] = {}
        user_ratings: dict[int, dict[int, float]] = {}
        user_movies: dict[int, set[int]] = {}

        for uid, mid, r in ratings:
            movie_ratings.setdefault(mid, {})[uid] = r
            user_ratings.setdefault(uid, {})[mid] = r
            user_movies.setdefault(uid, set()).add(mid)

        movie_norms = {mid: _l2_norm_sparse(v) for mid, v in movie_ratings.items()}

        movie_avg_rating: dict[int, float] = {}
        movie_rating_count: dict[int, int] = {}
        for mid, by_user in movie_ratings.items():
            n = len(by_user)
            if n <= 0:
                continue
            movie_rating_count[mid] = n
            movie_avg_rating[mid] = sum(by_user.values()) / float(n)

        docs = {mid: movie_text(m) for mid, m in movies.items()}
        tfidf = fit_tfidf(docs)

        return cls(
            _Indexes(
                movies=movies,
                movie_ratings=movie_ratings,
                movie_avg_rating=movie_avg_rating,
                movie_rating_count=movie_rating_count,
                movie_norms=movie_norms,
                user_ratings=user_ratings,
                user_movies=user_movies,
                tfidf=tfidf,
            )
        )

    def movie_title(self, movie_id: int) -> str:
        m = self._idx.movies.get(movie_id)
        return m.title if m else f"<unknown movieId={movie_id}>"

    def movie_genres(self, movie_id: int) -> str:
        m = self._idx.movies.get(movie_id)
        return m.genres if m else ""

    def movie_avg_rating(self, movie_id: int) -> float | None:
        return self._idx.movie_avg_rating.get(movie_id)

    def movie_rating_count(self, movie_id: int) -> int:
        return self._idx.movie_rating_count.get(movie_id, 0)

    def _find_movies_by_title(self, query: str) -> list[int]:
        q = normalize_text(query)
        if not q:
            return []
        hits: list[int] = []
        for mid, m in self._idx.movies.items():
            if q in normalize_text(m.title):
                hits.append(mid)
        hits.sort(key=lambda mid: self._idx.movies[mid].title)
        return hits
        
    
   

    @staticmethod
    def _base_title(title: str) -> str:
        # MovieLens titles are usually like: "Toy Story (1995)"
        t = title.strip()
        t = re.sub(r"\s*\(\d{4}\)\s*$", "", t)
        return normalize_text(t)

    def _resolve_title_to_movie_id(self, title_query: str) -> int | None:
        hits = self._find_movies_by_title(title_query)
        if not hits:
            return None
        if len(hits) == 1:
            return hits[0]

        # If the user typed an exact base title (ignoring year), prefer that.
        q_base = self._base_title(title_query)
        exact_base = [mid for mid in hits if self._base_title(self.movie_title(mid)) == q_base]
        if len(exact_base) == 1:
            return exact_base[0]

        print("Multiple matches found. Choose one:")
        for i, mid in enumerate(hits[:20], start=1):
            print(f"{i:>2}. {self.movie_title(mid)} (movieId={mid})")
        while True:
            raw = input(f"Enter 1..{min(20, len(hits))} (or blank to cancel): ").strip()
            if raw == "":
                return None
            try:
                choice = int(raw)
            except ValueError:
                continue
            if 1 <= choice <= min(20, len(hits)):
                return hits[choice - 1]

    def _collab_score_candidates(self, user_id: int) -> dict[int, float]:
        """
        Item-item collaborative scoring:
        score(candidate) = sum_{seen} sim(candidate, seen) * rating(user, seen)
        where sim is cosine similarity over sparse rating vectors (users -> rating).
        """
        seen = self._idx.user_ratings.get(user_id)
        if not seen:
            return {}

        scores: dict[int, float] = {}
        denom: dict[int, float] = {}

        for seen_mid, seen_rating in seen.items():
            # Candidate set: movies rated by users who rated seen_mid.
            raters = self._idx.movie_ratings.get(seen_mid, {})
            for rater_uid in raters.keys():
                for cand_mid, _ in self._idx.user_ratings.get(rater_uid, {}).items():
                    if cand_mid in seen:
                        continue
                    # accumulate similarity later; we compute sim on demand
                    # and keep a normalization denom so movies with more neighbors don't dominate
                    sim = self._movie_similarity_collab(seen_mid, cand_mid)
                    if sim <= 0.0:
                        continue
                    scores[cand_mid] = scores.get(cand_mid, 0.0) + sim * seen_rating
                    denom[cand_mid] = denom.get(cand_mid, 0.0) + abs(sim)

        for mid in list(scores.keys()):
            d = denom.get(mid, 0.0)
            if d > 0.0:
                scores[mid] /= d
            else:
                del scores[mid]

        return scores

    def _movie_similarity_collab(self, a_mid: int, b_mid: int) -> float:
        a = self._idx.movie_ratings.get(a_mid)
        b = self._idx.movie_ratings.get(b_mid)
        if not a or not b:
            return 0.0
        return cosine_sim_sparse(a, self._idx.movie_norms.get(a_mid, 0.0), b, self._idx.movie_norms.get(b_mid, 0.0))

    def _content_score_candidates(self, user_id: int, min_like_rating: float) -> dict[int, float]:
        """
        Builds a TF-IDF profile from movies the user liked, then scores unseen movies
        by cosine similarity to that profile.
        """
        seen = self._idx.user_ratings.get(user_id)
        if not seen:
            return {}

        liked = [(mid, (rating - min_like_rating + 1.0)) for mid, rating in seen.items() if rating >= min_like_rating]
        if not liked:
            return {}

        profile_vec, profile_norm = build_profile(self._idx.tfidf, liked)
        scores: dict[int, float] = {}
        for mid, vec in self._idx.tfidf.doc_vectors.items():
            if mid in seen:
                continue
            s = cosine_sim(profile_vec, profile_norm, vec, self._idx.tfidf.doc_norms.get(mid, 0.0))
            if s > 0.0:
                scores[mid] = s
        return scores

    @staticmethod
    def _minmax_norm(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        if hi <= lo:
            return {k: 0.0 for k in scores.keys()}
        inv = 1.0 / (hi - lo)
        return {k: (v - lo) * inv for k, v in scores.items()}

    def recommend_for_user(
        self,
        user_id: int,
        top_n: int = 10,
        method: str = "hybrid",
        min_like_rating: float = 4.0,
        alpha: float = 0.7,
    ) -> list[tuple[int, float]]:
        if method not in {"hybrid", "collab", "content"}:
            raise ValueError("method must be hybrid/collab/content")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")

        if method == "collab":
            scores = self._collab_score_candidates(user_id)
        elif method == "content":
            scores = self._content_score_candidates(user_id, min_like_rating=min_like_rating)
        else:
            collab = self._minmax_norm(self._collab_score_candidates(user_id))
            content = self._minmax_norm(self._content_score_candidates(user_id, min_like_rating=min_like_rating))
            keys = set(collab.keys()) | set(content.keys())
            scores = {k: alpha * collab.get(k, 0.0) + (1.0 - alpha) * content.get(k, 0.0) for k in keys}

        # Drop movies we don't have metadata for (shouldn't happen in MovieLens)
        scores = {mid: s for mid, s in scores.items() if mid in self._idx.movies}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[: max(0, int(top_n))]

    def similar_movies(
        self,
        title_query: str,
        top_n: int = 10,
        method: str = "content",
        alpha: float = 0.7,
    ) -> list[tuple[int, float]]:
        if method not in {"content", "collab", "hybrid"}:
            raise ValueError("method must be content/collab/hybrid")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")

        mid = self._resolve_title_to_movie_id(title_query)
        print(f"DEBUG mid={mid}, in tfidf={mid in self._idx.tfidf.doc_vectors}, in ratings={mid in self._idx.movie_ratings}")
        if mid is None:
            return []

        scores: dict[int, float] = {}
        if method in {"content", "hybrid"}:
            a = self._idx.tfidf.doc_vectors.get(mid, {})
            a_norm = self._idx.tfidf.doc_norms.get(mid, 0.0)
            content_scores: dict[int, float] = {}
            for other_mid, b in self._idx.tfidf.doc_vectors.items():
                if other_mid == mid:
                    continue
                s = cosine_sim(a, a_norm, b, self._idx.tfidf.doc_norms.get(other_mid, 0.0))
                if s > 0.0:
                    content_scores[other_mid] = s

        if method in {"collab", "hybrid"}:
            collab_scores: dict[int, float] = {}
            for other_mid in self._idx.movies.keys():
                if other_mid == mid:
                    continue
                s = self._movie_similarity_collab(mid, other_mid)
                if s > 0.0:
                    collab_scores[other_mid] = s

        if method == "content":
            scores = content_scores
        elif method == "collab":
            scores = collab_scores
        else:
            c_norm = self._minmax_norm(content_scores)
            k_norm = self._minmax_norm(collab_scores)
            keys = set(c_norm.keys()) | set(k_norm.keys())
            scores = {m: alpha * k_norm.get(m, 0.0) + (1.0 - alpha) * c_norm.get(m, 0.0) for m in keys}

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[: max(0, int(top_n))]

