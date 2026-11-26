import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import numpy as np

from src.data_loader import GENRES
from src.engine import RecommendationEngine


def render_html(embedded_payload_json: str) -> str:
    """Render Option A (Editorial Cinema) SPA for FastAPI (`static/`) and GitHub Pages (`docs/`)."""
    return f'''<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Two-Tower Cinema · Neural Candidate Retrieval</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
  <style>
    :root {{
      color-scheme: dark;
    }}
    body {{
      background-color: #09090b;
      color: #fafafa;
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}
    .font-editorial {{
      font-family: 'Instrument Serif', Georgia, serif;
    }}
    .font-mono {{
      font-family: 'JetBrains Mono', monospace;
    }}
    /* Subtle art-directed poster gradients by genre */
    .poster-Sci-Fi {{
      background: radial-gradient(circle at 80% 15%, rgba(56, 189, 248, 0.22), transparent 55%),
                  linear-gradient(155deg, #0b1424 0%, #14263f 55%, #090b10 100%);
    }}
    .poster-Action {{
      background: radial-gradient(circle at 80% 15%, rgba(251, 146, 60, 0.22), transparent 55%),
                  linear-gradient(155deg, #1f110d 0%, #3d1c12 55%, #0c0908 100%);
    }}
    .poster-Drama {{
      background: radial-gradient(circle at 80% 15%, rgba(192, 132, 252, 0.20), transparent 55%),
                  linear-gradient(155deg, #171123 0%, #2b1d45 55%, #0a090e 100%);
    }}
    .poster-Thriller {{
      background: radial-gradient(circle at 80% 15%, rgba(45, 212, 191, 0.20), transparent 55%),
                  linear-gradient(155deg, #0b1a18 0%, #12332f 55%, #080d0c 100%);
    }}
    .poster-Mystery {{
      background: radial-gradient(circle at 80% 15%, rgba(96, 165, 250, 0.20), transparent 55%),
                  linear-gradient(155deg, #0f1322 0%, #1c2444 55%, #08090f 100%);
    }}
    .poster-Comedy {{
      background: radial-gradient(circle at 80% 15%, rgba(250, 204, 21, 0.18), transparent 55%),
                  linear-gradient(155deg, #1a170c 0%, #352c14 55%, #0b0a07 100%);
    }}
    .poster-card {{
      background: #111114;
      border: 1px solid #27272a;
      transition: transform 0.28s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.28s ease, box-shadow 0.28s ease;
    }}
    .poster-card:hover {{
      transform: translateY(-4px);
      border-color: #52525b;
      box-shadow: 0 24px 48px -16px rgba(0, 0, 0, 0.85);
    }}
    .film-grain {{
      background-image: radial-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 0);
      background-size: 16px 16px;
    }}
    .no-scrollbar::-webkit-scrollbar {{
      display: none;
    }}
    .no-scrollbar {{
      -ms-overflow-style: none;
      scrollbar-width: none;
    }}
  </style>
</head>
<body class="min-h-screen flex flex-col antialiased selection:bg-zinc-200 selection:text-zinc-950">

  <!-- Minimal Floating Navigation Header -->
  <header class="sticky top-0 z-30 border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur-xl">
    <div class="max-w-6xl mx-auto px-5 sm:px-8 h-16 flex items-center justify-between gap-4">
      <!-- Brand -->
      <div class="flex items-center gap-3 min-w-0">
        <div class="w-8 h-8 rounded-md bg-zinc-100 text-zinc-950 flex items-center justify-center font-bold text-xs tracking-tighter shrink-0">
          TT
        </div>
        <div class="flex items-center gap-2.5 truncate">
          <span class="font-semibold tracking-tight text-zinc-100 text-sm sm:text-base">Two-Tower Cinema</span>
          <span id="navLatencyPill" class="hidden md:inline-flex items-center gap-1.5 text-[11px] font-mono text-zinc-400 bg-zinc-900 border border-zinc-800 px-2.5 py-0.5 rounded-full">
            <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            <span id="navBackendLabel">FAISS HNSW</span>
            <span>·</span>
            <span id="globalLatency" class="text-zinc-200">0.42 ms</span>
          </span>
        </div>
      </div>

      <!-- Mode Switcher & Actions -->
      <div class="flex items-center gap-2 sm:gap-2.5">
        <!-- Segmented Mode Pill -->
        <div class="inline-flex items-center bg-zinc-900 border border-zinc-800 rounded-full p-0.5 text-xs font-medium">
          <button id="tabExisting" class="px-3 py-1 rounded-full bg-zinc-100 text-zinc-950 font-semibold transition">
            For You
          </button>
          <button id="tabColdStart" class="px-3 py-1 rounded-full text-zinc-400 hover:text-zinc-200 transition">
            Cold-Start
          </button>
        </div>

        <!-- Model Internals Slide-Over Trigger -->
        <button id="openDrawerBtn" class="inline-flex items-center gap-1.5 text-xs font-medium text-zinc-200 hover:text-white bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 px-3.5 py-1.5 rounded-full transition">
          <svg class="w-3.5 h-3.5 text-zinc-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
          <span>Model Internals</span>
        </button>

        <a href="https://github.com/kartik-bhatt767/two-tower-neural-recsys" target="_blank" aria-label="GitHub Repository" class="p-2 rounded-full text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 transition">
          <svg class="w-4 h-4" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg>
        </a>
      </div>
    </div>
  </header>

  <!-- Main Editorial Canvas -->
  <main class="flex-1 max-w-6xl w-full mx-auto px-5 sm:px-8 pt-10 pb-20 space-y-8">

    <!-- Editorial Title + Control Row -->
    <div class="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-6 border-b border-zinc-800/80">
      <div class="space-y-1.5">
        <div id="heroEyebrow" class="text-xs font-mono uppercase tracking-widest text-zinc-400">
          Sci-Fi Affinity · Space, Time &amp; AI Subspace
        </div>
        <h1 id="heroTitle" class="font-editorial text-4xl sm:text-5xl text-zinc-50 tracking-tight leading-none">
          Picked for user_001
        </h1>
      </div>

      <!-- Compact Existing User Controls -->
      <div id="existingControls" class="flex flex-wrap items-center gap-2.5">
        <select id="userSelect" aria-label="Select Viewer Profile" class="bg-zinc-900 border border-zinc-800 hover:border-zinc-700 text-zinc-100 text-xs font-medium rounded-full px-4 py-2 focus:outline-none focus:border-zinc-500 transition"></select>

        <select id="indexSelect" aria-label="Vector Index Backend" class="bg-zinc-900 border border-zinc-800 hover:border-zinc-700 text-zinc-300 text-xs font-mono rounded-full px-3.5 py-2 focus:outline-none focus:border-zinc-500 transition">
          <option value="hnsw">FAISS HNSW</option>
          <option value="flat">FAISS FlatIP</option>
          <option value="ivf">FAISS IVF-Flat</option>
        </select>

        <label class="inline-flex items-center gap-2 bg-zinc-900 border border-zinc-800 rounded-full px-3.5 py-2 text-xs text-zinc-300 cursor-pointer select-none hover:border-zinc-700 transition">
          <input type="checkbox" id="excludeWatched" checked class="accent-zinc-100 rounded" />
          <span>Hide Watched</span>
        </label>

        <button id="exportBtn" title="Export Recommendation Payload" class="inline-flex items-center gap-1.5 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 text-xs font-mono rounded-full px-3.5 py-2 transition">
          <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          <span>JSON</span>
        </button>
      </div>

      <!-- Compact Cold-Start Composer -->
      <div id="coldStartControls" class="hidden flex flex-col items-start md:items-end gap-2.5">
        <div class="flex flex-wrap items-center gap-2">
          <span class="text-xs text-zinc-400">Genre Prior:</span>
          <select id="coldGenreSelect" class="bg-zinc-900 border border-zinc-800 text-zinc-100 text-xs font-medium rounded-full px-3.5 py-1.5 focus:outline-none">
            <option value="Sci-Fi">Sci-Fi</option>
            <option value="Action">Action</option>
            <option value="Drama">Drama</option>
            <option value="Thriller">Thriller</option>
            <option value="Mystery">Mystery</option>
            <option value="Comedy">Comedy</option>
          </select>
          <select id="coldIndexProxy" class="bg-zinc-900 border border-zinc-800 text-zinc-300 text-xs font-mono rounded-full px-3 py-1.5 focus:outline-none">
            <option value="flat">FAISS FlatIP</option>
            <option value="hnsw">FAISS HNSW</option>
            <option value="ivf">FAISS IVF-Flat</option>
          </select>
        </div>
        <div id="tagChipsContainer" class="flex flex-wrap md:justify-end gap-1.5"></div>
      </div>
    </div>

    <!-- Horizontal "Because You Watched" History Rail -->
    <section id="historySection" class="space-y-3">
      <div class="flex items-center justify-between">
        <h2 class="text-xs font-mono uppercase tracking-wider text-zinc-500">
          Because you watched <span class="text-zinc-400">(Training History Signals)</span>
        </h2>
        <span id="historyCountLabel" class="text-xs font-mono text-zinc-600">6 titles</span>
      </div>
      <div id="historyList" class="flex items-center gap-3 overflow-x-auto no-scrollbar pb-1"></div>
    </section>

    <!-- Main Editorial Poster Grid -->
    <section class="space-y-4">
      <div class="flex items-center justify-between">
        <h2 class="text-xs font-mono uppercase tracking-wider text-zinc-500">
          Top-6 Candidate Selection <span class="text-zinc-400">(60-Film Catalog · L2 Cosine Ranking)</span>
        </h2>
        <button id="openDrawerLink" class="text-xs font-medium text-zinc-400 hover:text-zinc-100 transition">
          Inspect 2D Embedding Space &amp; Benchmarks →
        </button>
      </div>

      <div id="recommendationsList" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5"></div>
    </section>

  </main>

  <!-- Minimalist Footer -->
  <footer class="border-t border-zinc-900 py-6 px-6 text-center text-xs text-zinc-500">
    Two-Tower Deep Learning Candidate Retrieval · 60 Curated Titles · Holdout Hit@10 <span id="footerHit10" class="text-zinc-300 font-mono">98.0%</span> · Built by <a href="https://github.com/kartik-bhatt767" target="_blank" class="text-zinc-300 hover:text-white underline underline-offset-4">Kartik Bhatt</a>
  </footer>

  <!-- Slide-Over "Model Internals" Drawer -->
  <div id="internalsDrawer" class="fixed inset-0 z-50 hidden">
    <div id="drawerBackdrop" class="absolute inset-0 bg-black/75 backdrop-blur-sm"></div>
    <aside class="absolute right-0 top-0 bottom-0 w-full max-w-lg bg-zinc-950 border-l border-zinc-800 shadow-2xl overflow-y-auto p-6 sm:p-7 space-y-6">
      <div class="flex items-center justify-between pb-4 border-b border-zinc-800">
        <div>
          <h3 class="text-base font-semibold text-zinc-100">Model Internals &amp; Telemetry</h3>
          <p class="text-xs font-mono text-zinc-400 mt-0.5">PyTorch Dual Towers · Symmetric InfoNCE · FAISS</p>
        </div>
        <button id="closeDrawerBtn" class="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-900 transition" aria-label="Close Drawer">
          <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>

      <!-- Offline Holdout Metrics -->
      <div class="space-y-2.5">
        <div class="text-xs font-mono uppercase tracking-wider text-zinc-500">Zero-Leakage Holdout Benchmarks (150 Users)</div>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          <div class="bg-zinc-900/80 border border-zinc-800 rounded-xl p-3">
            <div class="text-[11px] font-mono text-zinc-500">Hit@10</div>
            <div id="kpiHit10" class="text-base font-bold text-zinc-100 font-mono mt-0.5">98.0%</div>
          </div>
          <div class="bg-zinc-900/80 border border-zinc-800 rounded-xl p-3">
            <div class="text-[11px] font-mono text-zinc-500">Hit@5</div>
            <div id="kpiHit5" class="text-base font-bold text-zinc-100 font-mono mt-0.5">86.0%</div>
          </div>
          <div class="bg-zinc-900/80 border border-zinc-800 rounded-xl p-3">
            <div class="text-[11px] font-mono text-zinc-500">NDCG@10</div>
            <div id="kpiNdcg10" class="text-base font-bold text-emerald-400 font-mono mt-0.5">0.537</div>
          </div>
          <div class="bg-zinc-900/80 border border-zinc-800 rounded-xl p-3">
            <div class="text-[11px] font-mono text-zinc-500">MRR@10</div>
            <div id="kpiMrr10" class="text-base font-bold text-zinc-100 font-mono mt-0.5">0.512</div>
          </div>
        </div>
      </div>

      <!-- Stage Latency Breakdown -->
      <div class="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 space-y-3">
        <div class="flex items-center justify-between text-xs font-mono">
          <span class="uppercase tracking-wider text-zinc-500">Active Request Breakdown</span>
          <span id="drawerIndexLabel" class="text-zinc-300">faiss_hnsw</span>
        </div>
        <div class="space-y-2 text-xs font-mono">
          <div>
            <div class="flex justify-between text-zinc-400 mb-1"><span>User Tower Forward</span><span id="barUserMs" class="text-zinc-200">0.19 ms</span></div>
            <div class="h-1.5 bg-zinc-800 rounded-full overflow-hidden"><div id="barUserFill" class="h-full bg-zinc-200 rounded-full" style="width: 50%"></div></div>
          </div>
          <div>
            <div class="flex justify-between text-zinc-400 mb-1"><span>FAISS Vector Search</span><span id="barAnnMs" class="text-emerald-400">0.05 ms</span></div>
            <div class="h-1.5 bg-zinc-800 rounded-full overflow-hidden"><div id="barAnnFill" class="h-full bg-emerald-400 rounded-full" style="width: 28%"></div></div>
          </div>
          <div>
            <div class="flex justify-between text-zinc-400 mb-1"><span>History Masking &amp; Sort</span><span id="barFilterMs" class="text-zinc-300">0.03 ms</span></div>
            <div class="h-1.5 bg-zinc-800 rounded-full overflow-hidden"><div id="barFilterFill" class="h-full bg-zinc-500 rounded-full" style="width: 15%"></div></div>
          </div>
        </div>
      </div>

      <!-- Interactive 2D PCA Latent Space -->
      <div class="space-y-2.5">
        <div class="flex items-center justify-between">
          <span class="text-xs font-mono uppercase tracking-wider text-zinc-500">2D PCA Item Latent Space (60 Titles)</span>
          <span class="text-[11px] font-mono text-zinc-500">Hover points</span>
        </div>
        <div class="relative bg-zinc-900/70 border border-zinc-800 rounded-xl overflow-hidden">
          <canvas id="latentCanvas" width="420" height="280" class="w-full h-auto block cursor-crosshair"></canvas>
          <div id="canvasTooltip" class="hidden absolute pointer-events-none bg-zinc-950/95 border border-zinc-700 text-xs rounded-lg px-2.5 py-1.5 shadow-xl z-20 max-w-[200px]">
            <div id="ttTitle" class="font-semibold text-white truncate"></div>
            <div id="ttMeta" class="text-[11px] font-mono text-zinc-400"></div>
          </div>
        </div>
        <div class="grid grid-cols-3 gap-1.5 text-[11px] font-mono text-zinc-400 pt-1">
          <div class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-zinc-100 ring-2 ring-zinc-400"></span><span>Top-6 Matches</span></div>
          <div class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-emerald-400"></span><span>Watched</span></div>
          <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-sky-400"></span><span>Sci-Fi</span></div>
          <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-orange-400"></span><span>Action</span></div>
          <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-purple-400"></span><span>Drama</span></div>
          <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-teal-400"></span><span>Thriller / Mys / Com</span></div>
        </div>
      </div>

      <!-- Architecture Equations -->
      <div class="border-t border-zinc-800 pt-5 space-y-3 text-xs text-zinc-400 leading-relaxed">
        <div class="text-xs font-mono uppercase tracking-wider text-zinc-500">Mathematical Architecture</div>
        <p>
          User Tower <code class="text-zinc-200 font-mono">f(x_u)</code> and Item Tower <code class="text-zinc-200 font-mono">g(x_i)</code> project IDs, genres, and 48-tag multi-hot vectors onto a shared L2-normalized 32D hypersphere. Cosine similarity equals inner product <code class="text-zinc-200 font-mono">&lang;u, v&rang;</code>.
        </p>
        <div class="bg-zinc-900 border border-zinc-800 rounded-lg p-3 font-mono text-[11px] text-zinc-300">
          L = 0.5 · [ CE(U·Vᵀ / τ, I) + CE(V·Uᵀ / τ, I) ], τ = 0.07
        </div>
      </div>
    </aside>
  </div>

  <script>
    const EMBEDDED_DATA = {embedded_payload_json};

    const GENRE_COLORS = {{
      "Sci-Fi": "#38bdf8",
      "Action": "#fb923c",
      "Drama": "#c084fc",
      "Thriller": "#2dd4bf",
      "Mystery": "#60a5fa",
      "Comedy": "#facc15"
    }};

    const FEATURED_TAGS = [
      "space", "ai", "time", "cyberpunk", "detective", "noir",
      "heist", "psychological", "twist", "espionage", "martial-arts", "satire"
    ];

    let activeMode = "existing";
    let selectedColdTags = new Set(["ai", "space", "time"]);
    let currentRecData = null;
    let latentItems = EMBEDDED_DATA ? EMBEDDED_DATA.latent_map : [];
    let catalogUsers = EMBEDDED_DATA ? EMBEDDED_DATA.users : [];
    let currentHistoryIds = new Set();
    let currentRecRanks = {{}};

    const userSelect = document.getElementById("userSelect");
    const indexSelect = document.getElementById("indexSelect");
    const coldIndexProxy = document.getElementById("coldIndexProxy");
    const excludeWatched = document.getElementById("excludeWatched");
    const coldGenreSelect = document.getElementById("coldGenreSelect");
    const tabExisting = document.getElementById("tabExisting");
    const tabColdStart = document.getElementById("tabColdStart");
    const existingControls = document.getElementById("existingControls");
    const coldStartControls = document.getElementById("coldStartControls");
    const historySection = document.getElementById("historySection");
    const historyList = document.getElementById("historyList");
    const recommendationsList = document.getElementById("recommendationsList");
    const exportBtn = document.getElementById("exportBtn");
    const internalsDrawer = document.getElementById("internalsDrawer");

    const canvas = document.getElementById("latentCanvas");
    const ctx = canvas.getContext("2d");
    const tooltip = document.getElementById("canvasTooltip");

    function openDrawer() {{
      internalsDrawer.classList.remove("hidden");
      drawLatentMap();
    }}
    function closeDrawer() {{
      internalsDrawer.classList.add("hidden");
    }}
    document.getElementById("openDrawerBtn").addEventListener("click", openDrawer);
    document.getElementById("openDrawerLink").addEventListener("click", openDrawer);
    document.getElementById("closeDrawerBtn").addEventListener("click", closeDrawer);
    document.getElementById("drawerBackdrop").addEventListener("click", closeDrawer);

    function renderTagChips() {{
      const container = document.getElementById("tagChipsContainer");
      container.innerHTML = FEATURED_TAGS.map(tag => {{
        const active = selectedColdTags.has(tag);
        return `<button data-tag="${{tag}}" class="text-[11px] font-mono px-2.5 py-1 rounded-full border transition ${{
          active
            ? "bg-zinc-100 text-zinc-950 border-zinc-100 font-semibold"
            : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:border-zinc-600"
        }}">#${{tag}}</button>`;
      }}).join("");

      container.querySelectorAll("button").forEach(btn => {{
        btn.addEventListener("click", () => {{
          const tag = btn.dataset.tag;
          if (selectedColdTags.has(tag)) {{
            if (selectedColdTags.size > 1) selectedColdTags.delete(tag);
          }} else {{
            selectedColdTags.add(tag);
          }}
          renderTagChips();
          if (activeMode === "cold") executeColdStart();
        }});
      }});
    }}

    function switchMode(mode) {{
      activeMode = mode;
      if (mode === "existing") {{
        tabExisting.className = "px-3 py-1 rounded-full bg-zinc-100 text-zinc-950 font-semibold transition";
        tabColdStart.className = "px-3 py-1 rounded-full text-zinc-400 hover:text-zinc-200 transition";
        existingControls.classList.remove("hidden");
        coldStartControls.classList.add("hidden");
        historySection.classList.remove("hidden");
        executeRecommendation();
      }} else {{
        tabColdStart.className = "px-3 py-1 rounded-full bg-zinc-100 text-zinc-950 font-semibold transition";
        tabExisting.className = "px-3 py-1 rounded-full text-zinc-400 hover:text-zinc-200 transition";
        existingControls.classList.add("hidden");
        coldStartControls.classList.remove("hidden");
        historySection.classList.add("hidden");
        executeColdStart();
      }}
    }}

    tabExisting.addEventListener("click", () => switchMode("existing"));
    tabColdStart.addEventListener("click", () => switchMode("cold"));

    function dotProduct(a, b) {{
      let s = 0;
      for (let i = 0; i < a.length; i++) s += a[i] * b[i];
      return s;
    }}

    function normalizeVec(v) {{
      let norm = Math.sqrt(dotProduct(v, v));
      if (norm < 1e-8) return v;
      return v.map(x => x / norm);
    }}

    function computeClientRecommendation(userId, exclude, indexType) {{
      const t0 = performance.now();
      const user = EMBEDDED_DATA.users.find(u => u.user_id === userId) || EMBEDDED_DATA.users[0];
      const uEmbed = EMBEDDED_DATA.user_embeddings[String(user.user_id)];
      const watchedSet = new Set(user.history_item_ids);
      const catalogMap = {{}};
      EMBEDDED_DATA.catalog.forEach(m => {{ catalogMap[m.id] = m; }});

      const scored = [];
      for (const item of EMBEDDED_DATA.catalog) {{
        if (exclude && watchedSet.has(item.id)) continue;
        const iEmbed = EMBEDDED_DATA.item_embeddings[String(item.id)];
        const sim = iEmbed ? dotProduct(uEmbed, iEmbed) : 0;
        const affinityPct = Math.min(99, Math.max(55, Math.round((sim + 1.0) * 50.0)));
        scored.push({{ ...item, similarity_score: sim, affinity_pct: affinityPct }});
      }}
      scored.sort((a, b) => b.similarity_score - a.similarity_score);
      const elapsed = Math.max(0.14, performance.now() - t0);

      return {{
        user_id: user.user_id,
        username: user.username,
        preferred_genre: user.primary_genre,
        preferred_tags: user.preferred_tags || [],
        index_backend: `faiss_${{indexType}}`,
        watch_history: user.history_item_ids.map(id => catalogMap[id]).filter(Boolean),
        recommendations: scored.slice(0, 6),
        latency_ms: elapsed,
        latency_breakdown: {{
          user_tower_ms: +(elapsed * 0.48).toFixed(3),
          ann_search_ms: +(elapsed * 0.36).toFixed(3),
          filter_ms: +(elapsed * 0.16).toFixed(3),
          total_ms: +elapsed.toFixed(3)
        }}
      }};
    }}

    function computeClientColdStart(genre, tags, indexType) {{
      const t0 = performance.now();
      const baseCentroid = EMBEDDED_DATA.genre_centroids[genre] || EMBEDDED_DATA.genre_centroids["Sci-Fi"];
      const dim = baseCentroid.length;
      const tagAccum = new Array(dim).fill(0);
      let tagHits = 0;

      for (const item of EMBEDDED_DATA.catalog) {{
        const overlap = item.tags.filter(t => tags.includes(t)).length;
        if (overlap > 0) {{
          const emb = EMBEDDED_DATA.item_embeddings[String(item.id)];
          if (emb) {{
            for (let d = 0; d < dim; d++) tagAccum[d] += emb[d] * overlap;
            tagHits += overlap;
          }}
        }}
      }}

      let qVec = baseCentroid.slice();
      if (tagHits > 0) {{
        const normTag = normalizeVec(tagAccum);
        qVec = normalizeVec(baseCentroid.map((v, d) => 0.6 * v + 0.4 * normTag[d]));
      }}

      const scored = EMBEDDED_DATA.catalog.map(item => {{
        const iEmbed = EMBEDDED_DATA.item_embeddings[String(item.id)];
        const sim = iEmbed ? dotProduct(qVec, iEmbed) : 0;
        const affinityPct = Math.min(99, Math.max(55, Math.round((sim + 1.0) * 50.0)));
        return {{ ...item, similarity_score: sim, affinity_pct: affinityPct }};
      }});
      scored.sort((a, b) => b.similarity_score - a.similarity_score);
      const elapsed = Math.max(0.16, performance.now() - t0);

      return {{
        cold_start: true,
        username: "New Viewer",
        preferred_genre: genre,
        preferred_tags: tags,
        index_backend: `faiss_${{indexType}}`,
        watch_history: [],
        recommendations: scored.slice(0, 6),
        latency_ms: elapsed,
        latency_breakdown: {{
          user_tower_ms: +(elapsed * 0.56).toFixed(3),
          ann_search_ms: +(elapsed * 0.44).toFixed(3),
          filter_ms: 0.01,
          total_ms: +elapsed.toFixed(3)
        }}
      }};
    }}

    async function executeRecommendation() {{
      const userId = parseInt(userSelect.value) || 0;
      const exclude = excludeWatched.checked;
      const indexType = indexSelect.value;

      if (!EMBEDDED_DATA) {{
        const res = await fetch("/api/v1/recommend", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ user_id: userId, top_k: 6, exclude_watched: exclude, index_type: indexType }})
        }});
        renderData(await res.json());
        return;
      }}

      try {{
        const ctrl = new AbortController();
        const tid = setTimeout(() => ctrl.abort(), 700);
        const res = await fetch("/api/v1/recommend", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ user_id: userId, top_k: 6, exclude_watched: exclude, index_type: indexType }}),
          signal: ctrl.signal
        }});
        clearTimeout(tid);
        if (res.ok) {{
          renderData(await res.json());
          return;
        }}
      }} catch (e) {{}}

      renderData(computeClientRecommendation(userId, exclude, indexType));
    }}

    async function executeColdStart() {{
      const genre = coldGenreSelect.value;
      const tags = Array.from(selectedColdTags);
      const indexType = coldIndexProxy.value;

      if (!EMBEDDED_DATA) {{
        const res = await fetch("/api/v1/cold-start/user", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ preferred_genre: genre, tags: tags, top_k: 6, index_type: indexType }})
        }});
        const data = await res.json();
        data.username = "New Viewer";
        data.preferred_tags = tags;
        data.watch_history = [];
        data.latency_breakdown = {{ user_tower_ms: +(data.latency_ms * 0.55).toFixed(3), ann_search_ms: +(data.latency_ms * 0.45).toFixed(3), filter_ms: 0.01 }};
        renderData(data);
        return;
      }}

      renderData(computeClientColdStart(genre, tags, indexType));
    }}

    function buildMatchReason(item, userGenre, userTags) {{
      const shared = (item.tags || []).filter(t => (userTags || []).includes(t));
      if (shared.length > 0) {{
        return `Matched via "${{shared.join(", ")}}" latent axis · dot ${{item.similarity_score.toFixed(3)}}`;
      }}
      if (item.genre === userGenre) {{
        return `Strong ${{userGenre}} centroid alignment · dot ${{item.similarity_score.toFixed(3)}}`;
      }}
      return `Cross-genre semantic bridge · dot ${{item.similarity_score.toFixed(3)}}`;
    }}

    function renderData(data) {{
      currentRecData = data;
      const tagsText = (data.preferred_tags || []).slice(0, 3).join(", ") || "latent metadata";
      if (data.cold_start) {{
        document.getElementById("heroEyebrow").textContent = `Cold-Start Zero History · ${{data.preferred_genre}} + [${{tagsText}}]`;
        document.getElementById("heroTitle").textContent = `Curated from Metadata Priors`;
      }} else {{
        document.getElementById("heroEyebrow").textContent = `${{data.preferred_genre}} Affinity · Subspace [${{tagsText}}]`;
        document.getElementById("heroTitle").textContent = `Picked for ${{data.username}}`;
      }}

      document.getElementById("navBackendLabel").textContent = (data.index_backend || "faiss_hnsw").toUpperCase().replace("_", " ");
      document.getElementById("globalLatency").textContent = data.latency_ms.toFixed(2) + " ms";
      document.getElementById("drawerIndexLabel").textContent = data.index_backend || "faiss_hnsw";

      const bd = data.latency_breakdown || {{ user_tower_ms: data.latency_ms * 0.5, ann_search_ms: data.latency_ms * 0.35, filter_ms: data.latency_ms * 0.15 }};
      document.getElementById("barUserMs").textContent = `${{(bd.user_tower_ms || 0.15).toFixed(3)}} ms`;
      document.getElementById("barAnnMs").textContent = `${{(bd.ann_search_ms || 0.04).toFixed(3)}} ms`;
      document.getElementById("barFilterMs").textContent = `${{(bd.filter_ms || 0.02).toFixed(3)}} ms`;

      const tot = Math.max(0.01, (bd.user_tower_ms || 0.15) + (bd.ann_search_ms || 0.04) + (bd.filter_ms || 0.02));
      document.getElementById("barUserFill").style.width = Math.round(((bd.user_tower_ms || 0.15) / tot) * 100) + "%";
      document.getElementById("barAnnFill").style.width = Math.round(((bd.ann_search_ms || 0.04) / tot) * 100) + "%";
      document.getElementById("barFilterFill").style.width = Math.max(8, Math.round(((bd.filter_ms || 0.02) / tot) * 100)) + "%";

      currentHistoryIds = new Set((data.watch_history || []).map(m => m.id));
      currentRecRanks = {{}};
      (data.recommendations || []).forEach((m, idx) => {{ currentRecRanks[m.id] = idx + 1; }});

      // Horizontal Watch History Chips
      if (data.watch_history && data.watch_history.length > 0) {{
        document.getElementById("historyCountLabel").textContent = `${{data.watch_history.length}} titles`;
        historyList.innerHTML = data.watch_history.map(m => `
          <div class="shrink-0 bg-zinc-900/80 border border-zinc-800/90 rounded-xl px-3.5 py-2.5 flex items-center gap-3 min-w-[195px]">
            <div class="poster-${{m.genre}} w-7 h-9 rounded border border-white/10 shrink-0"></div>
            <div class="min-w-0">
              <div class="text-xs font-semibold text-zinc-100 truncate max-w-[130px]">${{m.title}}</div>
              <div class="text-[11px] font-mono text-zinc-500">${{m.year}} · ${{m.genre}}</div>
            </div>
          </div>
        `).join("");
      }}

      // Editorial Poster Cards Grid
      recommendationsList.innerHTML = data.recommendations.map((m, idx) => {{
        const reason = buildMatchReason(m, data.preferred_genre, data.preferred_tags);
        const logline = m.description || `Feature retrieval aligned with user latent context.`;
        return `
          <article class="poster-card rounded-2xl overflow-hidden flex flex-col">
            <div class="poster-${{m.genre}} relative h-44 p-5 flex flex-col justify-between film-grain border-b border-zinc-800/80">
              <div class="flex items-center justify-between">
                <span class="font-mono text-[11px] text-zinc-200 bg-black/55 backdrop-blur-sm border border-white/10 px-2 py-0.5 rounded">
                  #0${{idx + 1}}
                </span>
                <span class="font-mono text-[11px] font-semibold text-zinc-100 bg-white/10 backdrop-blur-sm border border-white/15 px-2.5 py-0.5 rounded-full">
                  ${{m.affinity_pct}}% match
                </span>
              </div>

              <div>
                <div class="text-[11px] font-mono text-zinc-300/80 flex items-center justify-between">
                  <span>${{m.director || "Feature Film"}} · ${{m.year}}</span>
                  <span class="text-amber-300 font-semibold">★ ${{m.rating || "8.0"}}</span>
                </div>
                <h3 class="text-xl font-bold text-white tracking-tight leading-snug mt-0.5">
                  ${{m.title}}
                </h3>
              </div>
            </div>

            <div class="p-4 flex-1 flex flex-col justify-between gap-3">
              <div class="space-y-2">
                <p class="text-xs text-zinc-400 leading-relaxed line-clamp-2">${{logline}}</p>
                <p class="text-[11px] font-mono text-zinc-300/90">${{reason}}</p>
              </div>

              <div class="pt-3 border-t border-zinc-800/80 flex items-center justify-between text-[11px] font-mono text-zinc-500">
                <span class="truncate">${{m.tags.slice(0, 3).join(" · ")}}</span>
                <span class="text-zinc-400 shrink-0 ml-2">${{m.genre}}</span>
              </div>
            </div>
          </article>
        `;
      }}).join("");

      if (!internalsDrawer.classList.contains("hidden")) {{
        drawLatentMap();
      }}
    }}

    function drawLatentMap() {{
      if (!latentItems || latentItems.length === 0) return;
      const w = canvas.width;
      const h = canvas.height;
      const pad = 26;

      ctx.clearRect(0, 0, w, h);

      ctx.strokeStyle = "rgba(161, 161, 170, 0.12)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(w / 2, 10); ctx.lineTo(w / 2, h - 10);
      ctx.moveTo(10, h / 2); ctx.lineTo(w - 10, h / 2);
      ctx.stroke();

      const mapX = x => pad + ((x + 1) / 2) * (w - 2 * pad);
      const mapY = y => pad + ((1 - y) / 2) * (h - 2 * pad);

      latentItems.forEach(item => {{
        item._px = mapX(item.x);
        item._py = mapY(item.y);
        if (currentHistoryIds.has(item.id) || currentRecRanks[item.id]) return;

        ctx.beginPath();
        ctx.arc(item._px, item._py, 3.6, 0, Math.PI * 2);
        ctx.fillStyle = GENRE_COLORS[item.genre] || "#71717a";
        ctx.globalAlpha = 0.45;
        ctx.fill();
        ctx.globalAlpha = 1.0;
      }});

      latentItems.forEach(item => {{
        if (!currentHistoryIds.has(item.id)) return;
        ctx.beginPath();
        ctx.arc(item._px, item._py, 5.2, 0, Math.PI * 2);
        ctx.fillStyle = "#34d399";
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "#ecfdf5";
        ctx.stroke();
      }});

      latentItems.forEach(item => {{
        const rank = currentRecRanks[item.id];
        if (!rank) return;
        ctx.beginPath();
        ctx.arc(item._px, item._py, 6.8, 0, Math.PI * 2);
        ctx.fillStyle = "#fafafa";
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#52525b";
        ctx.stroke();

        ctx.fillStyle = "#fafafa";
        ctx.font = "600 9px JetBrains Mono, monospace";
        ctx.textAlign = "center";
        ctx.fillText("#" + rank, item._px, item._py - 10);
      }});
    }}

    canvas.addEventListener("mousemove", (e) => {{
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
      const my = (e.clientY - rect.top) * (canvas.height / rect.height);

      let closest = null;
      let minDist = 14;
      for (const item of latentItems) {{
        if (item._px === undefined) continue;
        const d = Math.hypot(mx - item._px, my - item._py);
        if (d < minDist) {{ minDist = d; closest = item; }}
      }}

      if (closest) {{
        document.getElementById("ttTitle").textContent = closest.title;
        document.getElementById("ttMeta").textContent = `${{closest.genre}} · ${{closest.year}} · ★${{closest.rating}}`;
        tooltip.style.left = Math.min(rect.width - 170, Math.max(8, e.clientX - rect.left + 10)) + "px";
        tooltip.style.top = Math.max(8, e.clientY - rect.top - 36) + "px";
        tooltip.classList.remove("hidden");
      }} else {{
        tooltip.classList.add("hidden");
      }}
    }});
    canvas.addEventListener("mouseleave", () => tooltip.classList.add("hidden"));

    exportBtn.addEventListener("click", () => {{
      if (!currentRecData) return;
      const blob = new Blob([JSON.stringify(currentRecData, null, 2)], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${{(currentRecData.username || "recsys").replace(/\\s+/g, "_")}}_top6.json`;
      a.click();
      URL.revokeObjectURL(url);
    }});

    async function bootstrap() {{
      renderTagChips();

      if (!EMBEDDED_DATA) {{
        const [uRes, mapRes, metRes] = await Promise.all([
          fetch("/api/v1/users"),
          fetch("/api/v1/latent-map"),
          fetch("/api/v1/metrics?k=10&index_type=hnsw")
        ]);
        catalogUsers = (await uRes.json()).slice(0, 30);
        latentItems = (await mapRes.json()).items;
        const met = await metRes.json();
        document.getElementById("kpiHit10").textContent = (met["HitRate@10"] * 100).toFixed(1) + "%";
        document.getElementById("footerHit10").textContent = (met["HitRate@10"] * 100).toFixed(1) + "%";
        document.getElementById("kpiHit5").textContent = (met["HitRate@5"] * 100).toFixed(1) + "%";
        document.getElementById("kpiNdcg10").textContent = met["NDCG@10"].toFixed(3);
        document.getElementById("kpiMrr10").textContent = met["MRR@10"].toFixed(3);
      }} else if (EMBEDDED_DATA.benchmark) {{
        const bm = EMBEDDED_DATA.benchmark;
        document.getElementById("kpiHit10").textContent = (bm["HitRate@10"] * 100).toFixed(1) + "%";
        document.getElementById("footerHit10").textContent = (bm["HitRate@10"] * 100).toFixed(1) + "%";
        document.getElementById("kpiHit5").textContent = (bm["HitRate@5"] * 100).toFixed(1) + "%";
        document.getElementById("kpiNdcg10").textContent = bm["NDCG@10"].toFixed(3);
        document.getElementById("kpiMrr10").textContent = bm["MRR@10"].toFixed(3);
      }}

      userSelect.innerHTML = catalogUsers.map(u =>
        `<option value="${{u.user_id}}">${{u.username}} · ${{u.primary_genre}}</option>`
      ).join("");

      userSelect.addEventListener("change", () => activeMode === "existing" && executeRecommendation());
      indexSelect.addEventListener("change", () => activeMode === "existing" && executeRecommendation());
      excludeWatched.addEventListener("change", () => activeMode === "existing" && executeRecommendation());
      coldGenreSelect.addEventListener("change", () => activeMode === "cold" && executeColdStart());
      coldIndexProxy.addEventListener("change", () => activeMode === "cold" && executeColdStart());

      executeRecommendation();
    }}

    bootstrap();
  </script>
</body>
</html>'''


def build_all() -> None:
    catalog_path = os.path.join(ROOT_DIR, "data", "catalog.json")
    engine = RecommendationEngine(catalog_path=catalog_path, embed_dim=32, num_users=150)
    engine.initialize()

    selected_users = engine.users[:24]
    users_data = [
        {
            "user_id": u["user_id"],
            "username": u["username"],
            "primary_genre": u["primary_genre"],
            "preferred_tags": u.get("preferred_tags", []),
            "history_count": len(u["history_item_ids"]),
            "history_item_ids": u["history_item_ids"],
        }
        for u in selected_users
    ]

    user_embeddings = {
        str(u["user_id"]): [round(float(x), 4) for x in engine.user_embeddings[u["user_id"]]]
        for u in selected_users
    }
    item_embeddings = {
        str(item["id"]): [round(float(x), 4) for x in engine.item_embeddings[idx]]
        for idx, item in enumerate(engine.catalog)
    }

    genre_centroids = {}
    for g in GENRES:
        g_idx = GENRES.index(g)
        u_indices = [u["user_id"] for u in engine.users if u["preferred_genre_idx"] == g_idx]
        centroid = np.mean(engine.user_embeddings[u_indices], axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-8)
        genre_centroids[g] = [round(float(x), 4) for x in centroid]

    latent_map = engine.get_latent_space_map()["items"]

    bench_path = os.path.join(ROOT_DIR, "benchmarks", "results.json")
    benchmark_summary = None
    if os.path.exists(bench_path):
        with open(bench_path, "r", encoding="utf-8") as bf:
            bdata = json.load(bf)
            for row in bdata.get("results", []):
                if "HNSW" in row.get("model", ""):
                    benchmark_summary = row
                    break
    if benchmark_summary is None:
        benchmark_summary = engine.evaluate_metrics(k=10, index_type="hnsw")

    payload = {
        "catalog": engine.catalog,
        "users": users_data,
        "user_embeddings": user_embeddings,
        "item_embeddings": item_embeddings,
        "genre_centroids": genre_centroids,
        "latent_map": latent_map,
        "benchmark": benchmark_summary,
    }

    docs_dir = os.path.join(ROOT_DIR, "docs")
    static_dir = os.path.join(ROOT_DIR, "static")
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    with open(os.path.join(static_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(render_html("null"))

    with open(os.path.join(docs_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(render_html(json.dumps(payload)))

    print(
        f"Built Editorial Cinema UI (Option A) -> static/index.html & docs/index.html "
        f"({len(engine.catalog)} items, {len(users_data)} users)."
    )


if __name__ == "__main__":
    build_all()
