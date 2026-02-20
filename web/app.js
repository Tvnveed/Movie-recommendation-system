async function apiGet(path, params) {
  const url = new URL(path, window.location.origin);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      url.searchParams.set(k, String(v));
    }
  }
  const res = await fetch(url.toString(), { method: "GET" });
  const json = await res.json().catch(() => null);
  if (!res.ok) {
    const msg = (json && json.error) ? json.error : `Request failed (${res.status})`;
    throw new Error(msg);
  }
  if (!json || json.ok !== true) {
    throw new Error("Unexpected response from server");
  }
  return json;
}

function el(id) {
  return document.getElementById(id);
}

function setPill(ok, text) {
  const pill = el("healthPill");
  pill.textContent = text;
  pill.classList.remove("ok", "bad");
  pill.classList.add(ok ? "ok" : "bad");
}

function renderResults(targetId, items) {
  const ol = el(targetId);
  ol.innerHTML = "";
  for (const it of items) {
    const li = document.createElement("li");
    const row = document.createElement("div");
    row.className = "row";
    const title = document.createElement("div");
    title.className = "title";
    title.textContent = it.title ?? "(unknown)";
    const meta = document.createElement("div");
    meta.className = "meta";
    const score = typeof it.score === "number" ? it.score.toFixed(4) : String(it.score ?? "");
    meta.textContent = `movieId=${it.movieId} • score=${score}`;
    row.appendChild(title);
    row.appendChild(meta);
    li.appendChild(row);

    const reason = document.createElement("div");
    reason.className = "reason";
    const genres = (it.genres || "").split("|").filter(Boolean);
    const genresText = genres.length ? genres.join(", ") : "no genre info";
    const avg = typeof it.avgRating === "number" ? it.avgRating.toFixed(2) : "N/A";
    const num = typeof it.numRatings === "number" ? it.numRatings : 0;
    reason.textContent =
      `Why we picked this: It has similar genres (${genresText}) to your favourite movie. ` +
      `${num} people watched and rated it an average of ${avg} out of 5 stars. ` +
      `The more similar the genres and viewing patterns, the higher it ranks for you!`;
    li.appendChild(reason);

    ol.appendChild(li);
  }
}

async function checkHealth() {
  try {
    await apiGet("/api/health");
    setPill(true, "Server: OK");
  } catch (e) {
    setPill(false, "Server: not reachable");
  }
}

window.addEventListener("DOMContentLoaded", () => {
  checkHealth();

  const btnUserSimple = el("btnUserSimple");
  if (btnUserSimple) {
    btnUserSimple.addEventListener("click", (ev) => {
      ev.preventDefault();
      userFlowRecommend();
    });
  }

  // Live search dropdown
  const favoriteInput = el("favoriteMovie");
  const dropdown = el("movieDropdown");
  let debounceTimer;

  favoriteInput.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    const q = favoriteInput.value.trim();
    if (!q) { dropdown.style.display = "none"; return; }
    debounceTimer = setTimeout(async () => {
      try {
        const json = await apiGet("/api/search-movies", { q });
        dropdown.innerHTML = "";
        if (!json.items.length) { dropdown.style.display = "none"; return; }
        for (const item of json.items) {
          const div = document.createElement("div");
          div.className = "dropdown-item";
          div.textContent = item.title;
          div.addEventListener("click", () => {
            favoriteInput.value = item.title;
            dropdown.style.display = "none";
          });
          dropdown.appendChild(div);
        }
        dropdown.style.display = "block";
      } catch {}
    }, 300);
  });

  document.addEventListener("click", (e) => {
    if (!favoriteInput.contains(e.target) && !dropdown.contains(e.target)) {
      dropdown.style.display = "none";
    }
  });
});

async function userFlowRecommend() {
  el("errUserSimple").textContent = "";
  renderResults("resultsUserSimple", []);

  const name = el("userName").value.trim();
  if (!name) {
    el("errUserSimple").textContent = "Please enter your name.";
    return;
  }
  const favorite = el("favoriteMovie").value.trim();
  if (!favorite) {
    el("errUserSimple").textContent = "Please enter your favourite movie.";
    return;
  }
  const prefInput = document.querySelector('input[name="prefType"]:checked');
  const preference = prefInput ? prefInput.value : "similar";
  const topN = el("topNUserSimple").value || "10";
  const methodSelect = el("simMethod");
  const method = methodSelect ? methodSelect.value : "hybrid";

  try {
    const alpha = preference === "very-close" ? 0.85 : 0.5;
    const json = await apiGet("/api/similar-movies", {
      title: favorite,
      topN,
      method,
      alpha,
    });
    renderResults("resultsUserSimple", json.items || []);

    const titleEl = el("topPicksTitle");
    if (titleEl) {
      titleEl.classList.add("visible");
    }
    const expl = el("explanation");
    if (expl) {
      const prefText =
        preference === "very-close"
          ? "that stay very close to your favourite movie, using more weight on collaborative patterns."
          : "that are similar to your favourite movie but with more variety, balancing content and collaborative signals.";
      expl.textContent = `Because you love "${favorite}", we picked movies ${prefText}`;
    }
    el("errUserSimple").textContent = `Here are your recommendations, ${name}.`;
  } catch (e) {
    el("errUserSimple").textContent = e.message;
  }
}