from __future__ import annotations

import argparse
import json
import mimetypes
import posixpath
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from movie_recommender.data import ensure_movielens_small
from movie_recommender.recommender import MovieRecommender


def _json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _bad_request(handler: BaseHTTPRequestHandler, message: str) -> None:
    _json(handler, HTTPStatus.BAD_REQUEST, {"ok": False, "error": message})


def _internal_error(handler: BaseHTTPRequestHandler, message: str) -> None:
    _json(handler, HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": message})


class App:
    def __init__(self, data_dir: Path, web_dir: Path):
        self.data_dir = Path(data_dir)
        self.web_dir = Path(web_dir)
        dataset_dir = ensure_movielens_small(self.data_dir)
        self.recommender = MovieRecommender.from_movielens(dataset_dir)


def _safe_join(base: Path, url_path: str) -> Path | None:
    """
    Resolve url_path under base, preventing path traversal.
    """
    url_path = url_path.split("?", 1)[0].split("#", 1)[0]
    url_path = posixpath.normpath(url_path)
    url_path = url_path.lstrip("/")
    candidate = (base / url_path).resolve()
    try:
        base_resolved = base.resolve()
    except FileNotFoundError:
        base_resolved = base.resolve()
    if base_resolved not in candidate.parents and candidate != base_resolved:
        return None
    return candidate


class Handler(BaseHTTPRequestHandler):
    app: App  # injected

    def log_message(self, format: str, *args) -> None:  # noqa: A003 (format shadow)
        # keep console output clean; uncomment for debugging
        # super().log_message(format, *args)
        return

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self._handle_api(parsed)
                return
            self._handle_static(parsed.path)
        except Exception as e:
            _internal_error(self, str(e))

    def _handle_api(self, parsed) -> None:
        qs = parse_qs(parsed.query)
        path = parsed.path

        if path == "/api/health":
            _json(self, HTTPStatus.OK, {"ok": True})
            return

        if path == "/api/search-movies":
            query = qs.get("q", [""])[0].strip()
            if not query:
                _json(self, HTTPStatus.OK, {"ok": True, "items": []})
                return
            hits = self.app.recommender._find_movies_by_title(query)[:20]
            items = [
                {
                    "movieId": mid,
                    "title": self.app.recommender.movie_title(mid),
                }
                for mid in hits
            ]
            _json(self, HTTPStatus.OK, {"ok": True, "items": items})
            return

        if path == "/api/recommend-user":
            try:
                user_id = int(qs.get("userId", [""])[0])
            except ValueError:
                _bad_request(self, "userId must be an integer")
                return
            try:
                top_n = int(qs.get("topN", ["10"])[0])
            except ValueError:
                _bad_request(self, "topN must be an integer")
                return

            method = qs.get("method", ["hybrid"])[0]
            min_rating = float(qs.get("minRating", ["4.0"])[0])
            alpha = float(qs.get("alpha", ["0.7"])[0])

            recs = self.app.recommender.recommend_for_user(
                user_id=user_id,
                top_n=top_n,
                method=method,
                min_like_rating=min_rating,
                alpha=alpha,
            )
            items = [
                {
                    "movieId": mid,
                    "title": self.app.recommender.movie_title(mid),
                    "score": score,
                    "genres": self.app.recommender.movie_genres(mid),
                    "avgRating": self.app.recommender.movie_avg_rating(mid),
                    "numRatings": self.app.recommender.movie_rating_count(mid),
                }
                for mid, score in recs
            ]
            _json(self, HTTPStatus.OK, {"ok": True, "items": items})
            return

        if path == "/api/similar-movies":
            title = qs.get("title", [""])[0].strip()
            if not title:
                _bad_request(self, "title is required")
                return
            try:
                top_n = int(qs.get("topN", ["10"])[0])
            except ValueError:
                _bad_request(self, "topN must be an integer")
                return
            method = qs.get("method", ["content"])[0]
            alpha = float(qs.get("alpha", ["0.7"])[0])

            recs = self.app.recommender.similar_movies(
                title_query=title,
                top_n=top_n,
                method=method,
                alpha=alpha,
            )
            items = [
                {
                    "movieId": mid,
                    "title": self.app.recommender.movie_title(mid),
                    "score": score,
                    "genres": self.app.recommender.movie_genres(mid),
                    "avgRating": self.app.recommender.movie_avg_rating(mid),
                    "numRatings": self.app.recommender.movie_rating_count(mid),
                }
                for mid, score in recs
            ]
            _json(self, HTTPStatus.OK, {"ok": True, "items": items})
            return

        _json(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Unknown endpoint"})

    def _handle_static(self, path: str) -> None:
        if path in {"", "/"}:
            path = "/index.html"

        target = _safe_join(self.app.web_dir, path)
        if target is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if target.is_dir():
            target = target / "index.html"
        if not target.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        ctype, _ = mimetypes.guess_type(str(target))
        if not ctype:
            ctype = "application/octet-stream"
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    ap = argparse.ArgumentParser(description="Local web server for the movie recommender UI.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--web-dir", type=Path, default=Path("web"))
    args = ap.parse_args()

    app = App(data_dir=args.data_dir, web_dir=args.web_dir)

    Handler.app = app  # type: ignore[attr-defined]
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)

    print(f"Serving UI on http://{args.host}:{args.port}/", file=sys.stderr)
    print("Press Ctrl+C to stop.", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())