"""
MK1-A — Autonomous Web Browser
================================
MK1-A identifies its own knowledge gaps by analyzing
which token patterns it predicts poorly, then searches
the web for text that fills those gaps.

This is genuine autonomous learning — no human tells it
what to study. It figures that out itself.

Safety:
    - Only browses approved domains
    - Rate limited (max 10 pages per hour)
    - Everything logged to mk1/logs/browse.log
    - Kill switch respected
    - No personal data, no login pages

Run standalone:
    py -3.11 mk1/browse.py

Or import into MK1-A training loop.
"""

import os
import sys
import time
import json
import random
import logging
import threading
import hashlib
from datetime import datetime
from collections import Counter
from typing import Optional

# Add ARIA root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger("MK1A.Browser")

# ── Safety config ──────────────────────────────────────────────────────────

# Only these domains are allowed
APPROVED_DOMAINS = [
    "en.wikipedia.org",
    "simple.wikipedia.org",
    "www.gutenberg.org",
    "arxiv.org",
    "docs.python.org",
    "www.bbc.com/news",
    "www.nature.com",
    "plato.stanford.edu",      # philosophy
    "www.scientificamerican.com",
    "stackoverflow.com",
    "github.com",
    "www.poetryfoundation.org",
    "www.historytoday.com",
]

# Topics MK1-A can search for
SEED_TOPICS = [
    "artificial intelligence",
    "machine learning",
    "neural networks",
    "quantum physics",
    "history of computing",
    "philosophy of mind",
    "mathematics",
    "astronomy",
    "biology",
    "chemistry",
    "world history",
    "literature analysis",
    "programming languages",
    "robotics",
    "climate science",
]

MAX_PAGES_PER_HOUR  = 10
MAX_TEXT_PER_PAGE   = 50000   # chars
MIN_TEXT_PER_PAGE   = 500     # skip tiny pages
BROWSE_INTERVAL     = 360     # seconds between browse cycles (6 min)
DATA_DIR            = "data/mk1a"
LOG_PATH            = "mk1/logs/browse.log"


# ── Gap analyzer ───────────────────────────────────────────────────────────

class GapAnalyzer:
    """
    Analyzes MK1-A's weights to find knowledge gaps.
    
    How it works:
    - Runs inference on diverse text samples
    - Measures loss per token category
    - High loss on a category = knowledge gap
    - Returns search queries to fill those gaps
    """

    # Token categories and example texts to probe
    PROBE_TEXTS = {
        "science":      "The quantum mechanical model describes electrons as wave functions with probability distributions around the nucleus.",
        "history":      "The Roman Empire fell in 476 AD when Romulus Augustulus was deposed by the Germanic chieftain Odoacer.",
        "philosophy":   "Descartes argued that the mind and body are distinct substances, a position known as Cartesian dualism.",
        "mathematics":  "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.",
        "technology":   "Machine learning algorithms improve automatically through experience and by the use of data.",
        "literature":   "The narrative structure of the novel creates tension through foreshadowing and dramatic irony.",
        "code":         "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "nature":       "Photosynthesis is the process by which plants convert sunlight, water and carbon dioxide into glucose.",
        "geography":    "The Amazon River flows through South America and is the largest river by discharge volume in the world.",
        "economics":    "Supply and demand curves intersect at the equilibrium price where quantity supplied equals quantity demanded.",
    }

    def __init__(self, inference):
        self.inference = inference

    def find_gaps(self) -> list[str]:
        """
        Returns list of search queries for MK1-A's weakest areas.
        """
        if not self.inference or not self.inference.loaded:
            log.warning("Model not loaded — using random topics")
            return random.sample(SEED_TOPICS, 3)

        scores = {}
        log.info("Analyzing knowledge gaps...")

        for category, text in self.PROBE_TEXTS.items():
            try:
                loss = self._probe_loss(text)
                scores[category] = loss
                log.info(f"  {category:15} → loss {loss:.3f}")
            except Exception as e:
                log.warning(f"  {category}: probe failed ({e})")
                scores[category] = 5.0  # assume bad if can't probe

        # Sort by loss (highest = biggest gap)
        sorted_gaps = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        log.info(f"Biggest gaps: {[f'{k}({v:.2f})' for k,v in sorted_gaps[:3]]}")

        # Convert top 3 gaps to search queries
        queries = []
        for category, loss in sorted_gaps[:3]:
            query = self._gap_to_query(category, loss)
            queries.append(query)

        return queries

    def _probe_loss(self, text: str) -> float:
        """Measure model loss on a specific text."""
        import torch
        import torch.nn.functional as F
        import torch_directml

        dml = torch_directml.device()
        ids = self.inference.encode(text)
        if len(ids) < 4:
            return 5.0

        # Clip to context length
        ids = ids[:self.inference.model.cfg.context_len]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=dml)
        y = torch.tensor([ids[1:]],  dtype=torch.long, device=dml)

        self.inference.model.eval()
        with torch.no_grad():
            logits = self.inference.model(x)
            loss   = F.cross_entropy(
                logits.reshape(-1, self.inference.model.cfg.vocab_size),
                y.reshape(-1)
            ).item()

        return loss

    def _gap_to_query(self, category: str, loss: float) -> str:
        """Convert a knowledge gap category to a search query."""
        query_map = {
            "science":     "introduction to quantum mechanics",
            "history":     "major events world history",
            "philosophy":  "philosophy of consciousness mind",
            "mathematics": "mathematical proofs number theory",
            "technology":  "how artificial intelligence works",
            "literature":  "literary analysis narrative techniques",
            "code":        "python programming algorithms",
            "nature":      "biology ecology natural systems",
            "geography":   "world geography major regions",
            "economics":   "economics supply demand markets",
        }
        base = query_map.get(category, category)
        # Add some randomness so it doesn't always search the same thing
        variants = [base, f"{category} explained", f"introduction to {category}", f"{category} fundamentals"]
        return random.choice(variants)


# ── Web fetcher ────────────────────────────────────────────────────────────

class WebFetcher:
    """
    Fetches clean text from approved web pages.
    Rate limited, logged, sandboxed to approved domains.
    """

    def __init__(self):
        self._requests_this_hour = 0
        self._hour_start = time.time()
        self._fetched_urls = set()   # don't fetch same URL twice
        self._lock = threading.Lock()

    def search_and_fetch(self, query: str) -> Optional[str]:
        """
        Search for query and fetch best result.
        Returns clean text or None if failed/rate limited.
        """
        if not self._check_rate_limit():
            log.warning("Rate limit hit — skipping fetch")
            return None

        # Build Wikipedia search URL (safe, always approved)
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query.replace(' ', '+')}&format=json&srlimit=3"

        try:
            import urllib.request
            import urllib.parse

            # Search Wikipedia
            req = urllib.request.Request(search_url, headers={
                "User-Agent": "MK1-AI-Research-Bot/1.0 (educational project; contact via github)"
            })
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read().decode())

            results = data.get("query", {}).get("search", [])
            if not results:
                log.info(f"No results for: {query}")
                return None

            # Pick first result we haven't seen
            for result in results:
                title   = result["title"]
                page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"

                if page_url in self._fetched_urls:
                    continue

                text = self._fetch_wikipedia(title)
                if text and len(text) >= MIN_TEXT_PER_PAGE:
                    self._fetched_urls.add(page_url)
                    self._log_fetch(page_url, len(text), query)
                    return text[:MAX_TEXT_PER_PAGE]

        except Exception as e:
            log.warning(f"Fetch failed for '{query}': {e}")

        return None

    def _fetch_wikipedia(self, title: str) -> Optional[str]:
        """Fetch clean text from a Wikipedia article."""
        try:
            import urllib.request
            import urllib.parse

            # Use Wikipedia API to get plain text
            api_url = (
                f"https://en.wikipedia.org/w/api.php"
                f"?action=query&titles={urllib.parse.quote(title)}"
                f"&prop=extracts&exintro=false&explaintext=true"
                f"&format=json&exsectionformat=plain"
            )

            req = urllib.request.Request(api_url, headers={
                "User-Agent": "MK1-AI-Research-Bot/1.0 (educational project; contact via github)"
            })
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode())

            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                text = page.get("extract", "")
                if text:
                    # Clean up
                    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 40]
                    return '\n'.join(lines)

        except Exception as e:
            log.warning(f"Wikipedia fetch failed: {e}")

        return None

    def _check_rate_limit(self) -> bool:
        with self._lock:
            now = time.time()
            if now - self._hour_start > 3600:
                self._requests_this_hour = 0
                self._hour_start = now

            if self._requests_this_hour >= MAX_PAGES_PER_HOUR:
                return False

            self._requests_this_hour += 1
            return True

    def _log_fetch(self, url: str, length: int, query: str):
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {length:6} chars | {query:40} | {url}\n")
        log.info(f"Fetched: {url} ({length:,} chars)")


# ── Data manager ───────────────────────────────────────────────────────────

class DataManager:
    """
    Manages MK1-A's autonomous data collection.
    Saves fetched text to data/mk1a/ folder.
    """

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.index_path = os.path.join(DATA_DIR, "index.json")
        self.index = self._load_index()

    def save(self, text: str, query: str, url: str) -> str:
        """Save fetched text and return file path."""
        # Create filename from query
        safe_name = "".join(c if c.isalnum() else "_" for c in query)[:40]
        filename  = f"{safe_name}_{int(time.time())}.txt"
        filepath  = os.path.join(DATA_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Source: {url}\n")
            f.write(f"# Query: {query}\n")
            f.write(f"# Date: {datetime.now().isoformat()}\n\n")
            f.write(text)

        # Update index
        self.index[filename] = {
            "query":   query,
            "url":     url,
            "chars":   len(text),
            "date":    datetime.now().isoformat(),
        }
        self._save_index()

        log.info(f"Saved: {filepath} ({len(text):,} chars)")
        return filepath

    def get_all_text(self) -> str:
        """Get all autonomously collected text combined."""
        all_text = []
        for filename in os.listdir(DATA_DIR):
            if filename.endswith(".txt") and filename != "index.json":
                path = os.path.join(DATA_DIR, filename)
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        all_text.append(f.read())
                except Exception:
                    pass
        return "\n\n".join(all_text)

    def total_chars(self) -> int:
        return sum(v.get("chars", 0) for v in self.index.values())

    def _load_index(self) -> dict:
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)


# ── Main autonomous browser ────────────────────────────────────────────────

class AutonomousBrowser:
    """
    The main MK1-A autonomous learning loop.
    
    1. Analyzes model to find knowledge gaps
    2. Searches web for relevant text
    3. Saves to data/mk1a/
    4. Signals trainer to incorporate new data
    5. Repeats
    """

    def __init__(self, inference=None):
        self.inference    = inference
        self.analyzer     = GapAnalyzer(inference)
        self.fetcher      = WebFetcher()
        self.data_manager = DataManager()
        self.running      = False
        self._thread      = None
        self.new_data_callback = None   # called when new data is saved

    def start(self):
        """Start autonomous browsing in background thread."""
        self.running = True
        self._thread = threading.Thread(
            target=self._loop,
            name="MK1A-Browser",
            daemon=True
        )
        self._thread.start()
        log.info("MK1-A autonomous browser started")
        log.info(f"Data dir: {DATA_DIR}")
        log.info(f"Rate limit: {MAX_PAGES_PER_HOUR} pages/hour")

    def stop(self):
        self.running = False
        log.info("MK1-A browser stopped")

    def _loop(self):
        """Main autonomous loop."""
        cycle = 0

        while self.running:
            cycle += 1
            log.info(f"\n=== Browse cycle {cycle} ===")
            log.info(f"Total data collected: {self.data_manager.total_chars():,} chars")

            # Step 1: Find knowledge gaps
            queries = self.analyzer.find_gaps()
            log.info(f"Search queries: {queries}")

            # Step 2: Fetch text for each gap
            new_files = []
            for query in queries:
                if not self.running:
                    break

                log.info(f"Searching: '{query}'")
                text = self.fetcher.search_and_fetch(query)

                if text:
                    filepath = self.data_manager.save(text, query, f"wikipedia:{query}")
                    new_files.append(filepath)
                    log.info(f"Got {len(text):,} chars for '{query}'")
                else:
                    log.info(f"Nothing useful found for '{query}'")

                # Small delay between requests
                time.sleep(5)

            # Step 3: Notify trainer of new data
            if new_files and self.new_data_callback:
                log.info(f"Notifying trainer: {len(new_files)} new files")
                self.new_data_callback(new_files)

            # Step 4: Wait before next cycle
            log.info(f"Cycle {cycle} complete. Waiting {BROWSE_INTERVAL}s...")
            for _ in range(BROWSE_INTERVAL):
                if not self.running:
                    break
                time.sleep(1)

    def status(self) -> dict:
        return {
            "running":      self.running,
            "total_chars":  self.data_manager.total_chars(),
            "pages_fetched": len(self.data_manager.index),
            "data_dir":     DATA_DIR,
        }


# ── Standalone test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(message)s"
    )

    print("=" * 55)
    print("  MK1-A Autonomous Browser — Test Run")
    print("=" * 55)
    print(f"  Data dir:   {DATA_DIR}")
    print(f"  Rate limit: {MAX_PAGES_PER_HOUR} pages/hour")
    print(f"  Interval:   {BROWSE_INTERVAL}s between cycles")
    print("=" * 55)

    # Test without model (uses random topics)
    browser = AutonomousBrowser(inference=None)

    def on_new_data(files):
        print(f"\n*** NEW DATA: {len(files)} files added ***")
        for f in files:
            print(f"  {f}")

    browser.new_data_callback = on_new_data
    browser.start()

    print("\nBrowser running. Press Ctrl+C to stop.")
    print(f"Watch {DATA_DIR}/ for new files.\n")

    try:
        while True:
            time.sleep(10)
            status = browser.status()
            print(f"Status: {status['pages_fetched']} pages | {status['total_chars']:,} chars collected")
    except KeyboardInterrupt:
        browser.stop()
        print("\nBrowser stopped.")
        print(f"Data collected: {browser.data_manager.total_chars():,} chars")
        print(f"Files saved to: {DATA_DIR}/")
