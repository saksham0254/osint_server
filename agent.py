"""
openai_osint_agent.py
A ReAct agent with OpenAI function calling: scrape → embed/FAISS → optional Tor OSINT search.
"""

import os, re, json, time, requests, faiss, textwrap, numpy as np, sqlite3
from bs4 import BeautifulSoup, Comment
from datetime import datetime, timedelta
import openai
from pydantic import BaseModel, Field
from typing import List,Literal
import threading
import logging
import sys
from urllib.parse import urlparse
import urllib.parse
from dotenv import load_dotenv
import concurrent.futures

# Load environment variables from .env file
load_dotenv()
MAX_ATTEMPTS = 1

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- Globals ---------------------------------------------------------

CHAT_MODEL    = "gpt-4.1-nano"  # or "gpt-4-turbo"
VECTOR_DIM    = 1536
index         = faiss.IndexFlatIP(VECTOR_DIM)
id2meta       = {}        # idx → {"url":…, "title":…}


TOR_PROXIES = {"http": "socks5h://127.0.0.1:9050",
               "https": "socks5h://127.0.0.1:9050"}

VISITED_DB = 'webpages_state.db'

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required. Please set it in your .env file or environment.")

# Set the API key for OpenAI client
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Thread lock for database operations
db_lock = threading.Lock()
faiss_lock = threading.Lock()  # Lock for FAISS index operations

# Circuit breaker for search engines
class CircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_timeout=300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e

# Initialize circuit breakers for each search engine
search_breakers = {
    "tor66": CircuitBreaker(failure_threshold=2, recovery_timeout=90),
    "OSS": CircuitBreaker(failure_threshold=2, recovery_timeout=90),
    "Haystak": CircuitBreaker(failure_threshold=2, recovery_timeout=90),
    "Submarine": CircuitBreaker(failure_threshold=2, recovery_timeout=90),
    "onionland": CircuitBreaker(failure_threshold=2, recovery_timeout=60),
}

def init_db():
    """Initialize users database for authentication."""
    conn = sqlite3.connect(VISITED_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

def init_db_tables():
    """Initialize all database tables for thread-safe operations."""
    conn = sqlite3.connect(VISITED_DB)
    c = conn.cursor()
    
    # visited_urls table with failed status
    c.execute('''CREATE TABLE IF NOT EXISTS visited_urls (
        url TEXT PRIMARY KEY,
        visited INTEGER DEFAULT 0,
        failed INTEGER DEFAULT 0,
        failed_count INTEGER DEFAULT 0,
        last_failed TEXT,
        timestamp TEXT,
        last_updated TEXT,
        content TEXT
    )''')
    
    # faiss_metadata table with text content
    c.execute('''CREATE TABLE IF NOT EXISTS faiss_metadata (
        faiss_id INTEGER PRIMARY KEY,
        url TEXT,
        title TEXT,
        chunk_id INTEGER,
        text_content TEXT,
        timestamp TEXT
    )''')
    
    # faiss_vectors table (for storing embeddings)
    c.execute('''CREATE TABLE IF NOT EXISTS faiss_vectors (
        faiss_id INTEGER PRIMARY KEY,
        vector_data BLOB,
        timestamp TEXT
    )''')
    
    conn.commit()
    conn.close()

# Initialize both databases
init_db()
init_db_tables()

def store_faiss_vector(faiss_id: int, vector: list):
    """Store FAISS vector in database."""
    with db_lock:
        conn = sqlite3.connect(VISITED_DB)
        c = conn.cursor()
        now = datetime.utcnow().isoformat()
        vector_blob = np.array(vector, dtype="float32").tobytes()
        c.execute('''INSERT OR REPLACE INTO faiss_vectors 
                     (faiss_id, vector_data, timestamp) 
                     VALUES (?, ?, ?)''', 
                  (faiss_id, vector_blob, now))
        conn.commit()
        conn.close()

def validate_and_cleanup_faiss_consistency():
    """Validate and cleanup any inconsistencies between FAISS index and database."""
    global index
    
    with db_lock:
        conn = sqlite3.connect(VISITED_DB)
        c = conn.cursor()
        
        # Get all FAISS IDs from metadata
        c.execute('SELECT faiss_id FROM faiss_metadata ORDER BY faiss_id')
        metadata_ids = [row[0] for row in c.fetchall()]
        
        # Get all FAISS IDs from vectors
        c.execute('SELECT faiss_id FROM faiss_vectors ORDER BY faiss_id')
        vector_ids = [row[0] for row in c.fetchall()]
        
        conn.close()
    
    # Check for out-of-range IDs
    max_valid_id = index.ntotal - 1
    out_of_range_metadata = [id for id in metadata_ids if id > max_valid_id or id < 0]
    out_of_range_vectors = [id for id in vector_ids if id > max_valid_id or id < 0]
    
    # Check for missing vectors
    missing_vectors = set(metadata_ids) - set(vector_ids)
    
    # Check for orphaned vectors (vectors without metadata)
    orphaned_vectors = set(vector_ids) - set(metadata_ids)
    
    # Log issues found
    if out_of_range_metadata:
        logger.warning(f"Found {len(out_of_range_metadata)} out-of-range FAISS IDs in metadata: {out_of_range_metadata}")
    if out_of_range_vectors:
        logger.warning(f"Found {len(out_of_range_vectors)} out-of-range FAISS IDs in vectors: {out_of_range_vectors}")
    if missing_vectors:
        logger.warning(f"Found {len(missing_vectors)} metadata entries without vectors: {missing_vectors}")
    if orphaned_vectors:
        logger.warning(f"Found {len(orphaned_vectors)} vector entries without metadata: {orphaned_vectors}")
    
    # Clean up out-of-range and orphaned entries
    if out_of_range_metadata or out_of_range_vectors or missing_vectors or orphaned_vectors:
        logger.info("Cleaning up inconsistent FAISS entries...")
        
        with db_lock:
            conn = sqlite3.connect(VISITED_DB)
            c = conn.cursor()
            
            # Remove out-of-range metadata
            if out_of_range_metadata:
                placeholders = ','.join('?' * len(out_of_range_metadata))
                c.execute(f'DELETE FROM faiss_metadata WHERE faiss_id IN ({placeholders})', out_of_range_metadata)
                logger.info(f"Removed {len(out_of_range_metadata)} out-of-range metadata entries")
            
            # Remove out-of-range vectors
            if out_of_range_vectors:
                placeholders = ','.join('?' * len(out_of_range_vectors))
                c.execute(f'DELETE FROM faiss_vectors WHERE faiss_id IN ({placeholders})', out_of_range_vectors)
                logger.info(f"Removed {len(out_of_range_vectors)} out-of-range vector entries")
            
            # Remove metadata without vectors
            if missing_vectors:
                placeholders = ','.join('?' * len(missing_vectors))
                c.execute(f'DELETE FROM faiss_metadata WHERE faiss_id IN ({placeholders})', list(missing_vectors))
                logger.info(f"Removed {len(missing_vectors)} metadata entries without vectors")
            
            # Remove orphaned vectors
            if orphaned_vectors:
                placeholders = ','.join('?' * len(orphaned_vectors))
                c.execute(f'DELETE FROM faiss_vectors WHERE faiss_id IN ({placeholders})', list(orphaned_vectors))
                logger.info(f"Removed {len(orphaned_vectors)} orphaned vector entries")
            
            conn.commit()
            conn.close()
        
        logger.info("FAISS consistency cleanup completed")
    else:
        logger.info("FAISS index and database are consistent")

def load_faiss_index_from_db():
    """Load all vectors from database into FAISS index."""
    global index
    with faiss_lock:
        index = faiss.IndexFlatIP(VECTOR_DIM)  # Reset index
    
    with db_lock:
        conn = sqlite3.connect(VISITED_DB)
        c = conn.cursor()
        c.execute('SELECT faiss_id, vector_data FROM faiss_vectors ORDER BY faiss_id')
        rows = c.fetchall()
        conn.close()
    
    vectors = []
    for row in rows:
        faiss_id, vector_bytes = row
        vector = np.frombuffer(vector_bytes, dtype="float32")
        vectors.append(vector)
    
    if vectors:
        vectors_array = np.array(vectors, dtype="float32")
        with faiss_lock:
            index.add(vectors_array)
        print(f"Loaded {len(vectors)} vectors into FAISS index")
    
    # Validate and cleanup consistency after loading
    validate_and_cleanup_faiss_consistency()

def get_next_faiss_id() -> int:
    """Get the next available FAISS ID from database."""
    with db_lock:
        conn = sqlite3.connect(VISITED_DB)
        c = conn.cursor()
        c.execute('SELECT MAX(faiss_id) FROM faiss_metadata')
        result = c.fetchone()
        conn.close()
        return (result[0] or 0) + 1

def store_faiss_metadata(faiss_id: int, url: str, title: str, chunk_id: int, text_content: str = None):
    """Store FAISS metadata in database."""
    with db_lock:
        conn = sqlite3.connect(VISITED_DB)
        c = conn.cursor()
        now = datetime.utcnow().isoformat()
        c.execute('''INSERT OR REPLACE INTO faiss_metadata 
                     (faiss_id, url, title, chunk_id, text_content, timestamp) 
                     VALUES (?, ?, ?, ?, ?, ?)''', 
                  (faiss_id, url, title, chunk_id, text_content, now))
        conn.commit()
        conn.close()

def get_faiss_metadata(faiss_id: int) -> dict:
    """Get FAISS metadata from database."""
    with db_lock:
        conn = sqlite3.connect(VISITED_DB)
        c = conn.cursor()
        c.execute('SELECT url, title, chunk_id, text_content, timestamp FROM faiss_metadata WHERE faiss_id = ?', (faiss_id,))
        row = c.fetchone()
        conn.close()
        if row:
            return {
                "url": row[0], 
                "title": row[1], 
                "chunk_id": row[2],
                "text_content": row[3],
                "timestamp" : row[4]
            }
        return None

def is_url_visited_db(url: str) -> bool:
    """Check if URL is visited using database (thread-safe)."""
    with db_lock:
        conn = sqlite3.connect(VISITED_DB)
        c = conn.cursor()
        c.execute('SELECT visited FROM visited_urls WHERE url = ?', (url,))
        row = c.fetchone()
        conn.close()
        return row is not None and row[0] == 1

def is_url_failed_db(url: str) -> bool:
    """Check if URL is marked as failed using database (thread-safe)."""
    with db_lock:
        conn = sqlite3.connect(VISITED_DB)
        c = conn.cursor()
        c.execute('SELECT failed FROM visited_urls WHERE url = ?', (url,))
        row = c.fetchone()
        conn.close()
        return row is not None and row[0] == 1

def mark_url_visited_db(url: str, last_updated: str = None, content: str = None):
    """Mark URL as visited in database (thread-safe)."""
    with db_lock:
        conn = sqlite3.connect(VISITED_DB)
        c = conn.cursor()
        now = datetime.utcnow().isoformat()
        c.execute('''INSERT OR REPLACE INTO visited_urls 
                     (url, visited, failed, timestamp, last_updated, content) 
                     VALUES (?, 1, 0, ?, ?, ?)''', (url, now, last_updated, content))
        conn.commit()
        conn.close()

def mark_url_failed_db(url: str, max_failures: int = 3):
    """Mark URL as failed in database (thread-safe)."""
    with db_lock:
        conn = sqlite3.connect(VISITED_DB)
        c = conn.cursor()
        now = datetime.utcnow().isoformat()
        
        # Get current failed count
        c.execute('SELECT failed_count FROM visited_urls WHERE url = ?', (url,))
        row = c.fetchone()
        current_failures = (row[0] if row else 0) + 1
        
        if current_failures >= max_failures:
            # Mark as permanently failed
            c.execute('''INSERT OR REPLACE INTO visited_urls 
                         (url, visited, failed, failed_count, last_failed, timestamp) 
                         VALUES (?, 0, 1, ?, ?, ?)''', 
                      (url, current_failures, now, now))
            logger.warning(f"URL marked as permanently failed after {current_failures} attempts: {url}")
        else:
            # Increment failure count but don't mark as permanently failed yet
            c.execute('''INSERT OR REPLACE INTO visited_urls 
                         (url, visited, failed, failed_count, last_failed, timestamp) 
                         VALUES (?, 0, 0, ?, ?, ?)''', 
                      (url, current_failures, now, now))
            logger.info(f"URL failed attempt {current_failures}/{max_failures}: {url}")
        
        conn.commit()
        conn.close()

def add_links_to_db_threadsafe(links):
    """Add links to database as unvisited (thread-safe)."""
    with db_lock:
        conn = sqlite3.connect(VISITED_DB)
        c = conn.cursor()
        for url in links:
            c.execute('''INSERT OR IGNORE INTO visited_urls 
                         (url, visited, failed, failed_count, timestamp) 
                         VALUES (?, 0, 0, 0, NULL)''', (url,))
        conn.commit()
        conn.close()

def fetch_non_visited_url_db() -> str:
    """Fetch a non-visited, non-failed URL from database (thread-safe)."""
    with db_lock:
        conn = sqlite3.connect(VISITED_DB)
        c = conn.cursor()
        c.execute('''SELECT url FROM visited_urls 
                     WHERE visited=0 AND failed=0 
                     ORDER BY failed_count ASC, timestamp ASC''')
        row = c.fetchone()
        conn.close()
        return row[0] if row else None

# ---------- Tool 1: scrape ---------------------------------------------------

def chunk_text(text, chunk_size=1000, overlap=200):
    """Yield (chunk_id, chunk_text) with overlap."""
    start = 0
    chunk_id = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield chunk_id, text[start:end]
        chunk_id += 1
        start += chunk_size - overlap

def fetch_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:102.0) Gecko/20100101 Firefox/102.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'close',
    }
    try:
        response = requests.get(url, headers=headers, proxies=TOR_PROXIES, timeout=30)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                return response.text
    except Exception as e:
        return None
    return None

def extract_text(html: str) -> str:
    """
    Convert raw HTML to plain text.

    • Removes <script>, <style>, <noscript>, <iframe> blocks.
    • Drops HTML comments.
    • Collapses extra blank lines.
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1. Remove elements that never contain page copy
    for node in soup(["script", "style", "noscript", "iframe"]):
        node.decompose()

    # 2. Remove HTML comments
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    # 3. Extract & tidy
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)

def extract_links(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        url = requests.compat.urljoin(base_url, href)
        links.add(url)
    return links

def scrape_url(url: str) -> dict:
    """Scrape URL using database for visited tracking."""
    if is_url_visited_db(url):
        print(f"[visited-db] Already visited: {url}")
        raise Exception(f"URL already visited: {url}")
    print(f"[scrape] {url}")
    html = fetch_page(url)
    if not html:
        raise Exception(f"Failed to fetch page: {url}")
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title else url
    text = extract_text(html)
    # Extract last updated
    last_updated = None
    # Check HTTP header (if available in fetch_page)
    # last_updated = response.headers.get('Last-Modified', None)
    # Check meta tags
    meta_names = [
        {'name': 'last-modified'},
        {'property': 'article:modified_time'},
        {'http-equiv': 'last-modified'}
    ]
    for meta in meta_names:
        tag = soup.find('meta', attrs=meta)
        if tag and tag.get('content'):
            last_updated = tag['content']
            break
    # Extract and add links to DB as unvisited
    links = extract_links(html, url)
    add_links_to_db_threadsafe(links)
    mark_url_visited_db(url, last_updated, text)
    return {"url": url, "title": title, "text": text, "links": list(links), "last_updated": last_updated}

# ---------- Tool 2: index ----------------------------------------------------

client = openai.OpenAI()

def _get_embedding(text: str) -> list[float]:
    """
    Return a single 1536-dim embedding from the `text-embedding-3-small` model.
    Works with openai-python ≥ 1.0.
    """
    resp = client.embeddings.create(
        model   = "text-embedding-3-small",
        input   = text,
        encoding_format = "float"         # optional; float32 list (default)
    )
    return resp.data[0].embedding

def index_faiss(doc: dict) -> str:
    """Index document in FAISS with database persistence."""
    vec = _get_embedding(doc["text"])
    faiss_id = get_next_faiss_id()
    
    # Store in database
    store_faiss_metadata(faiss_id, doc["url"], doc["title"], doc.get("chunk_id", 0), doc["text"])
    store_faiss_vector(faiss_id, vec)
    
    # Add to FAISS index (thread-safe)
    with faiss_lock:
        index.add(np.array([vec], dtype="float32"))
    
    print(f"[index] added id {faiss_id}: {doc['url']} chunk {doc.get('chunk_id', 0)}")
    return str(faiss_id)

# ---------- Tool 3: Tor search ----------------------------------------------

def make_search_request(engine_name: str, url: str, timeout: int = 30) -> requests.Response:
    """Make a search request with circuit breaker protection."""
    print(f"[tor] making search request to {url}")
    def _make_request():
        return requests.get(
            url, 
            proxies=TOR_PROXIES, 
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:102.0) Gecko/20100101 Firefox/102.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
        )
    
    if engine_name in search_breakers:
        return search_breakers[engine_name].call(_make_request)
    else:
        return _make_request()

def tor_search(query: str, limit: int = 20) -> list[str]:
    """Search multiple dark web search engines for .onion links for a query."""
    print(f"[tor] search: {query!r}")
    
    # Multiple search engines for redundancy - fixed URLs and added alternatives
    search_engines = [
        {
            "name": "tor66",
            "url": f"http://tor66sewebgixwhcqfnp5inzp5x5uohhdy3kvtnyfxc2e5mxiuh34iid.onion/search?q={requests.utils.quote(query)}&sorttype=rel&page=1",
            "pattern": r"https?://[a-z2-7]{16,56}\.onion(?:/[^\s\"'<>]*)?"
        },
        {
            "name": "OSS",
            "url": f"http://3fzh7yuupdfyjhwt3ugzqqof6ulbcl27ecev33knxe3u7goi3vfn2qqd.onion/oss/?q={requests.utils.quote(query)}",
            "pattern": r"https?://[a-z2-7]{16,56}\.onion(?:/[^\s\"'<>]*)?"
        },
        {
            "name": "Haystak",
            "url": f"http://haystak5njsmn2hqkewecpaxetahtwhsbsa64jom2k22z5afxhnpxfid.onion/?q={requests.utils.quote(query)}",
            "pattern": r"https?://[a-z2-7]{16,56}\.onion(?:/[^\s\"'<>]*)?"
        },
        {
            "name": "Submarine",
            "url": f"http://no6m4wzdexe3auiupv2zwif7rm6qwxcyhslkcnzisxgeiw6pvjsgafad.onion/search.php?term={requests.utils.quote(query)}",
            "pattern": r"https?://[a-z2-7]{16,56}\.onion(?:/[^\s\"'<>]*)?"
        },
        {
            "name": "onionland",
            "url": f"http://3bbad7fauom4d6sgppalyqddsqbf5u5p56b5k5uk2zxsy3d6ey2jobad.onion/search?q={requests.utils.quote(query)}",
            "pattern": r"https?://[a-z2-7]{16,56}\.onion(?:/[^\s\"'<>]*)?"
        }
    ]
    
    def search_engine_worker(engine):
        try:
            logger.info(f"Trying {engine['name']}...")
            for attempt in range(MAX_ATTEMPTS):  # Try twice per engine
                try:
                    r = make_search_request(
                        engine['name'],
                        engine['url'], 
                        timeout=20
                    )
                    if r.status_code == 200:
                        links = re.findall(engine['pattern'], r.text)
                        if links:
                            logger.info(f"Found {len(links)} links from {engine['name']}")
                            return links
                        else:
                            logger.warning(f"No links found in {engine['name']} response")
                    else:
                        logger.warning(f"{engine['name']} returned status {r.status_code}")
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout on {engine['name']} attempt {attempt + 1}")
                    if attempt == 1:
                        continue
                    time.sleep(2)
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Request error on {engine['name']}: {e}")
                    if "SOCKS" in str(e) and attempt == 0:
                        logger.info("SOCKS error detected, attempting to renew Tor circuit...")
                        try:
                            renew_tor_circuit()
                            time.sleep(5)
                        except Exception as tor_e:
                            logger.warning(f"Failed to renew Tor circuit: {tor_e}")
                    break
                except Exception as e:
                    if "Circuit breaker is OPEN" in str(e):
                        logger.warning(f"Circuit breaker for {engine['name']} is OPEN, skipping")
                        break
                    else:
                        logger.error(f"Unexpected error with {engine['name']}: {e}")
                        break
        except Exception as e:
            logger.error(f"Unexpected error with {engine['name']}: {e}")
        return []

    # Run all engines in parallel
    all_links = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(search_engines)) as executor:
        future_to_engine = {executor.submit(search_engine_worker, engine): engine for engine in search_engines}
        for future in concurrent.futures.as_completed(future_to_engine):
            links = future.result()
            if links:
                all_links.extend(links)

    # Remove duplicates, filter valid URLs, and limit results
    unique_links = list(dict.fromkeys(all_links))
    valid_links = filter_valid_onion_urls(unique_links)[:limit]

    print("VALID LINKS:",valid_links)
    
    if valid_links:
        print(f"[tor] found {len(valid_links)} valid unique links from {len([e for e in search_engines if any(link in all_links for link in valid_links)])} engines")
    else:
        print(f"[tor] no valid links found from any search engine")
    
    return valid_links

def validate_onion_url(url: str) -> bool:
    """Validate if a URL is a proper .onion URL."""
    try:
        # Basic .onion URL validation
        if not url.startswith(('http://', 'https://')):
            return False
        
        # Extract domain
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Check if it's a .onion domain
        if not domain.endswith('.onion'):
            return False
        
        # Validate .onion domain format (base32 encoded, 16-56 chars)
        onion_part = domain[:-6]  # Remove '.onion'
        if not (16 <= len(onion_part) <= 56):
            return False
        
        # Check if it contains only valid base32 characters
        import re
        if not re.match(r'^[a-z2-7]+$', onion_part):
            return False
        
        return True
    except Exception:
        return False

def filter_valid_onion_urls(urls: list[str]) -> list[str]:
    """Filter a list of URLs to only include valid .onion URLs."""
    valid_urls = []
    for url in urls:
        if validate_onion_url(url):
            valid_urls.append(url)
        else:
            logger.debug(f"Filtered out invalid URL: {url}")
    return valid_urls

def call_llm(model, messages, response_format):
    completion = client.chat.completions.parse(
    model=model,
    messages=messages,
    response_format=response_format,
    )
    return json.loads(completion.choices[0].message.content)

class Response(BaseModel):
    intelligence_relevant : bool = Field(...,description = "If the information is relvant or not")
    confidence : int = Field(...,description = "A score from 0-100")
    category : List[Literal["cyber","malware","vulnerability","data breach","ransomware","phishing","physical","critical infra","cbrn","industrial safety","geopolitical","political","civil unrest","protest","disinformation","propaganda","extremist","terrorism","trafficking","human trafficking","weapons trafficking","drug trafficking","drugs","financial crime","fraud","money laundering","organized crime","military","military ops","weapons","espionage","insider threat","maritime","aviation","space","other"]]
    salient_phrases : List[str] = Field(...,description = 'up to 5 short snippets copied verbatim')
    region : str = Field(...,description = "Region of impact if mentioned; otheriwse none")
    rationale : str = Field(...,description = "1-2 line reasoning")


def condense_webpage(text):
    system_prompt = """You are OSINT-Guard, a security-intelligence classifier.

    TASK  
    ------
    Given the full plain-text of a web page, decide whether the page contains
    *actionable intelligence* about cyber or physical threats.  Security
    intelligence includes—but is not limited to—information on:

    • Vulnerabilities, exploits, malware, ransomware, zero-days  
    • Network or infrastructure reconnaissance, IP lists, C2 servers  
    • Illicit trafficking (weapons, drugs, human), smuggling routes  
    • Terrorism, extremist chatter, radicalisation content  
    • Operational security lapses (credentials, badge photos, floor plans)  
    • Insider threats, whistle-blown leaks, breach disclosures  
    • Plans or calls for violence, protests, sabotage  
    • Supply-chain & geopolitical risk affecting critical infrastructure

    It **excludes** purely journalistic summaries, general news with no
    actionable detail, marketing, tutorials with no malicious context, fiction,
    opinion pieces, or everyday crime reporting.

    OUTPUT FORMAT  
    -------------
    Return a single JSON object with these keys:

    ```json
    {
      "intelligence_relevant": true | false,      // boolean
      "confidence": 0-100,                        // subjective %
      "category": "<one of: cyber | physical | geopolitical | extremist | trafficking | other>",
      "region" : "region - country/city/state of impact otheriwse none"
      "rationale": "<2-3 sentences citing concrete phrases>",
      "salient_phrases": ["<up to 5 short snippets copied verbatim>"]
    }"""
    
    user_prompt = f"Webpage text: {text}"
    
    messages = [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}]
    model = 'gpt-4.1-nano'
    
    response = call_llm(model,messages,Response)
    return response
    
# ---------- Condensed intelligence storage -----------------------------------

def init_condensed_db():
    conn = sqlite3.connect(VISITED_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS condensed_pages (
        url TEXT PRIMARY KEY,
        timestamp TEXT,
        intelligence_relevant INTEGER,
        confidence INTEGER,
        category TEXT,
        salient_phrases TEXT,
        region TEXT,
        rationale TEXT
    )''')
    conn.commit()
    conn.close()

init_condensed_db()

def store_condensed_page(url: str, condensed: dict):
    """
    Store the output of condense_webpage for a given URL in the condensed_pages table.
    Converts lists to JSON strings for storage.
    """
    conn = sqlite3.connect(VISITED_DB)
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute('''INSERT OR REPLACE INTO condensed_pages
        (url, timestamp, intelligence_relevant, confidence, category, salient_phrases, region, rationale)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            url,
            now,
            int(condensed.get('intelligence_relevant', False)),
            int(condensed.get('confidence', 0)),
            json.dumps(condensed.get('category', [])),
            json.dumps(condensed.get('salient_phrases', [])),
            condensed.get('region', ''),
            condensed.get('rationale', '')
        )
    )
    conn.commit()
    conn.close()

def retrieve_condensed_pages(
    category: list = None,
    intelligence_relevant: int = None,
    confidence: int = None,
    timestamp: str = None
) -> list:
    """
    Retrieve condensed pages filtered by:
      - category: any match in list (if provided, SQL LIKE on JSON string)
      - intelligence_relevant: 0/1 (if provided)
      - confidence: minimum value (if provided)
      - timestamp: ISO string, only results after this (if provided)
    Returns a list of dicts.
    """
    conn = sqlite3.connect(VISITED_DB)
    c = conn.cursor()
    query = "SELECT url, timestamp, intelligence_relevant, confidence, category, salient_phrases, region, rationale FROM condensed_pages WHERE 1=1"
    params = []
    if intelligence_relevant is not None:
        query += " AND intelligence_relevant = ?"
        params.append(int(intelligence_relevant))
    if confidence is not None:
        query += " AND confidence >= ?"
        params.append(int(confidence))
    if timestamp is not None:
        query += " AND timestamp > ?"
        params.append(timestamp)
    if category:
        # Build a LIKE clause for each category in the list
        like_clauses = []
        for cat in category:
            like_clauses.append("category LIKE ?")
            params.append(f'%"{cat}"%')
        query += " AND (" + " OR ".join(like_clauses) + ")"
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    results = []
    for row in rows:
        cats = json.loads(row[4]) if row[4] else []
        results.append({
            "url": row[0],
            "timestamp": row[1],
            "intelligence_relevant": bool(row[2]),
            "confidence": row[3],
            "category": cats,
            "salient_phrases": json.loads(row[5]) if row[5] else [],
            "region": row[6],
            "rationale": row[7],
        })
    return results

def process_multiple_urls(max_urls: int = 5):
    """
    Master function to process multiple URLs:
      1. Scrape each URL (extract text, title, links)
      2. Add links to DB as unvisited
      3. Condense the webpage and store the result
      4. Chunk and index the text in FAISS
    Returns a dict with summary of all processed URLs.
    """
    
    processed_urls = []
    failed_urls = []
    
    for i in range(max_urls):
        url = fetch_non_visited_url_db()
        if url is None:
            break
        
        # Check if URL is already marked as failed
        if is_url_failed_db(url):
            logger.warning(f"Skipping failed URL: {url}")
            failed_urls.append({"url": url, "reason": "Already marked as failed"})
            continue
        
        scraped = None
        condensed = None
        chunk_ids = []
        
        try:
            # 1. Scrape
            scraped = scrape_url(url)
        except Exception as e:
            logger.error(f"Scrape failed for {url}: {e}")
            mark_url_failed_db(url)
            failed_urls.append({"url": url, "reason": f"Scrape failed: {e}"})
            continue
        
        try:
            # 2. Links are already added to DB by scrape_url
            # 3. Condense and store
            condensed = condense_webpage(scraped["text"])
            store_condensed_page(url, condensed)
        except Exception as e:
            logger.error(f"Condense/store failed for {url}: {e}")
            # Don't mark as failed for condensation errors, just log
            failed_urls.append({"url": url, "reason": f"Condense/store failed: {e}"})
            continue
        
        try:
            # 4. Chunk and index
            for chunk_id, chunk in chunk_text(scraped["text"]):
                doc = {"url": scraped["url"], "title": scraped["title"], "text": chunk, "chunk_id": chunk_id}
                idx = index_faiss(doc)
                chunk_ids.append(idx)
        except Exception as e:
            logger.error(f"Indexing failed for {url}: {e}")
            # Don't mark as failed for indexing errors, just log
            failed_urls.append({"url": url, "reason": f"Indexing failed: {e}"})
            continue
        
        processed_urls.append({
            "url": url,
            "title": scraped["title"],
            "chunks_indexed": len(chunk_ids),
            "intelligence_relevant": condensed.get("intelligence_relevant", False),
            "confidence": condensed.get("confidence", 0),
            "categories": condensed.get("category", [])
        })
    
    return {
        "processed_count": len(processed_urls),
        "failed_count": len(failed_urls),
        "processed_urls": processed_urls,
        "failed_urls": failed_urls,
        "summary": f"Processed {len(processed_urls)} URLs, {len(failed_urls)} failed"
    }

def process_url():
    """
    Legacy function that processes a single URL.
    Now calls process_multiple_urls with max_urls=1 for backward compatibility.
    """
    result = process_multiple_urls(max_urls=1)
    if result["processed_count"] == 0:
        return {"error": "No URLs processed"}
    
    # Return the first processed URL in the old format for compatibility
    processed = result["processed_urls"][0]
    return {
        "scraped": {"url": processed["url"], "title": processed["title"]},
        "condensed": {"intelligence_relevant": processed["intelligence_relevant"], "confidence": processed["confidence"]},
        "indexed_chunks": list(range(processed["chunks_indexed"]))
    }

# ---------- OpenAI ReAct agent with tools ------------------------------------

def add_links_to_unvisited(links):
    add_links_to_db_threadsafe(links)

def call_tool(name, arguments, context=None):
    if name == "process_url":
        return process_multiple_urls(max_urls=5)  # Process 5 URLs by default
    elif name == "tor_search":
        links = tor_search(**arguments)
        add_links_to_db_threadsafe(links)
        return {"links_added": links}
    else:
        return {"error": f"Unknown tool: {name}"}

def run_agent(n_steps: int = 30, seed_keyword = 'security threats'):
    tools = [
        {"type": "function", "function": {
            "name": "process_url",
            "description": "Process the next available unvisited URL: scrape it, extract links, condense the webpage, and index the text in FAISS. Returns scraped data, condensed intelligence summary, and indexed chunk IDs.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }},
        {"type": "function", "function": {
            "name": "tor_search",
            "description": "Search multiple dark web search engines for .onion links for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A **new and semantically differentv** search query/phrase based on the last webpage content."}
                },
                "required": ["query"],
            },
        }},
    ]

    start_msg = textwrap.dedent(f"""
        You are an OSINT agent. You can:
        - Scrape and process a URL (process_url)
        - If a processed page is security-relevant, search for new links (tor_search) and add them to the unvisited set. Generate a new search query/phrase based on the last webpage content.
        Repeat until there are no more unvisited URLs or you are told to stop.
        Start with the following keyword search on tor: {seed_keyword}
    """)

    messages = [
        {"role": "system", "content": start_msg},
    ]

    for step in range(n_steps):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="required",
        )
        assistant_msg = response.choices[0].message

        # Convert to dict for the messages list
        msg_dict = {
            "role": "assistant",
            "content": assistant_msg.content,
        }
        if hasattr(assistant_msg, "tool_calls") and assistant_msg.tool_calls:
            # tool_calls is a list of objects, convert each to dict
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in assistant_msg.tool_calls
            ]

        messages.append(msg_dict)

        if hasattr(assistant_msg, "tool_calls") and assistant_msg.tool_calls:
            for tool_call in assistant_msg.tool_calls:
                name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                tool_result = call_tool(name, arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": json.dumps(tool_result)
                })
        else:
            print("[Error]: Model did not return a tool call. Stopping.")
            break

class Category(BaseModel):
    category : List[Literal["cyber","malware","vulnerability","data breach","ransomware","phishing","physical","critical infra","cbrn","industrial safety","geopolitical","political","civil unrest","protest","disinformation","propaganda","extremist","terrorism","trafficking","human trafficking","weapons trafficking","drug trafficking","drugs","financial crime","fraud","money laundering","organized crime","military","military ops","weapons","espionage","insider threat","maritime","aviation","space","other"]]

def extract_category_from_query(query: str) -> list:
    """
    Use the LLM to extract relevant categories from a search query.
    Returns a list of category strings.
    """

    system_prompt = """You are an OSINT category extractor. Given a user query, return a list of relevant categories from this set:
    ["cyber","malware","vulnerability","data breach","ransomware","phishing","physical","critical infra","cbrn","industrial safety","geopolitical","political","civil unrest","protest","disinformation","propaganda","extremist","terrorism","trafficking","human trafficking","weapons trafficking","drug trafficking","drugs","financial crime","fraud","money laundering","organized crime","military","military ops","weapons","espionage","insider threat","maritime","aviation","space","other"]
    Return a JSON list of categories."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    resp = call_llm(CHAT_MODEL,messages,Category)
    cats = resp['category']
    return cats

def search_by_category_and_semantics(query: str, top_k: int = 20, sim_threshold: float = 0.5) -> list:
    """Search using database for metadata lookup with robust FAISS error handling."""
    
    categories = extract_category_from_query(query)
    if not categories:
        return []
    matched_pages = retrieve_condensed_pages(category=categories, intelligence_relevant=1, confidence=50)
    seen = set()
    unique_pages = []
    for page in matched_pages:
        if page["url"] not in seen:
            unique_pages.append(page)
            seen.add(page["url"])
    if not unique_pages:
        return []

    # Get FAISS IDs for these URLs from database (multiple IDs per URL due to chunking)
    url_to_ids = {}
    with db_lock:
        conn = sqlite3.connect(VISITED_DB)
        c = conn.cursor()
        for url in seen:
            c.execute('SELECT faiss_id FROM faiss_metadata WHERE url = ?', (url,))
            rows = c.fetchall()
            faiss_ids = [row[0] for row in rows]
            if faiss_ids:
                url_to_ids[url] = faiss_ids
        conn.close()
    
    if not url_to_ids:
        return []
    
    # Get all FAISS IDs we want to search in (flatten the lists)
    target_faiss_ids = []
    for faiss_ids in url_to_ids.values():
        target_faiss_ids.extend(faiss_ids)
    
    # Create a subset index with only the vectors from target URLs using index.reconstruct
    subset_index = faiss.IndexFlatIP(VECTOR_DIM)
    id_mapping = {}  # Map from subset index position to original FAISS ID
    with faiss_lock:
        for i, faiss_id in enumerate(target_faiss_ids):
            try:
                # Strict bounds check
                if not (0 <= faiss_id < index.ntotal):
                    logging.warning(f"Skipping invalid FAISS ID: {faiss_id} (index.ntotal={index.ntotal})")
                    continue
                vector = index.reconstruct(faiss_id)
                if vector is not None and len(vector) == VECTOR_DIM:
                    subset_index.add(np.array([vector], dtype="float32"))
                    id_mapping[i] = faiss_id
                else:
                    logging.warning(f"Invalid vector for FAISS ID {faiss_id}")
            except Exception as e:
                logging.warning(f"Failed to reconstruct vector for FAISS ID {faiss_id}: {e}")
                continue  # skip if vector not found
    
    if subset_index.ntotal == 0:
        return []
    
    query_vec = _get_embedding(query)
    with faiss_lock:
        D, I = subset_index.search(np.array([query_vec], dtype="float32"), subset_index.ntotal)

    results = []
    url_best_scores = {}  # Track best score per URL
    
    for i, (score, subset_idx) in enumerate(zip(D[0], I[0])):
        # Map back to original FAISS ID
        if subset_idx not in id_mapping:
            continue
        original_faiss_id = id_mapping[subset_idx]
        meta = get_faiss_metadata(original_faiss_id)
        if meta and meta["url"] in url_to_ids and score > sim_threshold:
            url = meta["url"]
            
            # Keep only the best score for each URL
            if url not in url_best_scores or score > url_best_scores[url]["similarity_score"]:
                condensed_data = None
                for page in matched_pages:
                    if page["url"] == url:
                        condensed_data = page
                        break
                
                result = {
                    "similarity_score": float(score),
                    "url": meta["url"],
                    "title": meta.get("title", ""),
                    "chunk_id": meta.get("chunk_id", 0),
                    "text_content": meta.get("text_content", ""),
                    "timestamp": meta.get("timestamp", "")
                }
                if condensed_data:
                    result["salient_phrases"] = condensed_data.get("salient_phrases", [])
                    result["category"] = condensed_data.get("category", [])
                    result["confidence"] = condensed_data.get("confidence", 0)
                    result["region"] = condensed_data.get("region", "")
                    result["rationale"] = condensed_data.get("rationale", "")
                
                url_best_scores[url] = result
    
    # Convert to list and sort by similarity score (descending)
    results = list(url_best_scores.values())
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Limit to top_k results
    return results[:top_k]

def get_category_counts() -> dict:
    """
    Return counts of URLs for each category in the condensed_pages database.
    Returns a dict with category names as keys and counts as values.
    """
    conn = sqlite3.connect(VISITED_DB)
    c = conn.cursor()
    c.execute('SELECT category FROM condensed_pages WHERE intelligence_relevant = 1 and confidence >= 50')
    rows = c.fetchall()
    conn.close()
    
    category_counts = {}
    for row in rows:
        try:
            categories = json.loads(row[0]) if row[0] else []
            for category in categories:
                category_counts[category] = category_counts.get(category, 0) + 1
        except (json.JSONDecodeError, TypeError):
            continue
    
    return category_counts

def generate_dynamic_keyword() -> str:
    """
    Use LLM to generate a new seed keyword based on:
    - Current category distribution
    - Underrepresented categories
    - Trending security topics
    """
    # Get current category counts
    category_counts = {}

    for c in ["cyber","malware","vulnerability","data breach","ransomware","phishing","physical","critical infra","cbrn","industrial safety","geopolitical","political","civil unrest","protest","disinformation","propaganda","extremist","terrorism","trafficking","human trafficking","weapons trafficking","drug trafficking","drugs","financial crime","fraud","money laundering","organized crime","military","military ops","weapons","espionage","insider threat","maritime","aviation","space","other"]:
        category_counts[c] = 0

    cc = get_category_counts()
    
    category_counts.update(cc) 
    
    print(category_counts)
    
    # Find underrepresented categories (less than 5 URLs)
    underrepresented = [cat for cat, count in category_counts.items() if count < 5]
    print(underrepresented)
    
    # Create prompt for keyword generation
    system_prompt = """You are an OSINT search phrase generator in 2-3 words. Based on the current data distribution and security landscape, generate a single, specific search phrase for web scraping.

    Available categories: ["cyber","malware","vulnerability","data breach","ransomware","phishing","physical","critical infra","cbrn","industrial safety","geopolitical","political","civil unrest","protest","disinformation","propaganda","extremist","terrorism","trafficking","human trafficking","weapons trafficking","drug trafficking","drugs","financial crime","fraud","money laundering","organized crime","military","military ops","weapons","espionage","insider threat","maritime","aviation","space","other"]

    Current category distribution: {category_counts}
    Underrepresented categories: {underrepresented}

    Generate a specific, actionable search phrase that would help find intelligence in underrepresented areas. Return only the phrase, nothing else. **The phrase doesn't have to one of the categories above.**"""

    user_prompt = f"Generate a search phrase for OSINT collection focusing on gaps in current data."
    
    messages = [
        {"role": "system", "content": system_prompt.format(
            category_counts=category_counts,
            underrepresented=underrepresented
        )},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.7
        )
        print(response)
        keyword = response.choices[0].message.content.strip().lower()
        # Clean up the keyword (remove quotes, extra text)
        keyword = keyword.replace('"', '').replace("'", "")
        print(keyword)
        logger.info(f"Generated new keyword: {keyword}")
        return keyword
    except Exception as e:
        logger.error(f"Error generating keyword: {e}")
        # Fallback to predefined keywords
        fallback_keywords = ["cyber", "terrorism", "weapons", "drugs", "fraud", "espionage"]
        return fallback_keywords[len(category_counts) % len(fallback_keywords)]

class PerpetualAgent:
    def __init__(self, delay_seconds: int = 60, max_steps_per_run: int = 20, dynamic_keywords: bool = True):
        self.delay_seconds = delay_seconds
        self.max_steps_per_run = max_steps_per_run
        self.dynamic_keywords = dynamic_keywords
        self.running = False
        self.last_run = None
        self.total_runs = 0
        self.total_urls_processed = 0
        self.current_keyword = "security threats"  # Default starting keyword
        
    def get_next_keyword(self) -> str:
        """Get the next keyword to use for the agent run."""
        if self.dynamic_keywords:
            self.current_keyword = generate_dynamic_keyword()
        return self.current_keyword
        
    def start(self):
        """Start the perpetual agent process."""
        self.running = True
        logger.info("Starting perpetual agent process...")
        
        while self.running:
            try:
                # Generate new keyword for this run
                keyword = self.get_next_keyword()
                logger.info(f"Starting agent run #{self.total_runs + 1} with keyword: {keyword}")
                start_time = datetime.now()
                
                # Run the agent with dynamic keyword
                run_agent(n_steps=self.max_steps_per_run, seed_keyword=keyword)
                
                self.total_runs += 1
                self.last_run = datetime.now()
                duration = (self.last_run - start_time).total_seconds()
                
                logger.info(f"Agent run #{self.total_runs} completed in {duration:.2f}s")
                logger.info(f"Waiting {self.delay_seconds}s before next run...")
                
                # Wait before next run
                time.sleep(self.delay_seconds)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping perpetual agent...")
                self.stop()
                break
            except Exception as e:
                logger.error(f"Error in perpetual agent run: {e}")
                logger.info(f"Waiting {self.delay_seconds}s before retry...")
                time.sleep(self.delay_seconds)
    
    def stop(self):
        """Stop the perpetual agent process."""
        self.running = False
        logger.info("Perpetual agent stopped.")
    
    def get_status(self) -> dict:
        """Get current status of the perpetual agent."""
        return {
            "running": self.running,
            "total_runs": self.total_runs,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "current_keyword": self.current_keyword,
            "delay_seconds": self.delay_seconds,
            "max_steps_per_run": self.max_steps_per_run,
            "dynamic_keywords": self.dynamic_keywords
        }

def run_perpetual_agent(delay_seconds: int = 60, max_steps_per_run: int = 20, dynamic_keywords: bool = True):
    """
    Start a perpetual agent that runs continuously with dynamic keyword generation.
    
    Args:
        delay_seconds: Time to wait between agent runs
        max_steps_per_run: Maximum steps per agent run
        dynamic_keywords: Whether to use LLM for keyword generation
    """
    agent = PerpetualAgent(
        delay_seconds=delay_seconds, 
        max_steps_per_run=max_steps_per_run,
        dynamic_keywords=dynamic_keywords
    )
    agent.start()

def get_circuit_breaker_status() -> dict:
    """Get the status of all circuit breakers for monitoring."""
    status = {}
    for name, breaker in search_breakers.items():
        status[name] = {
            "state": breaker.state,
            "failure_count": breaker.failure_count,
            "last_failure_time": breaker.last_failure_time,
            "failure_threshold": breaker.failure_threshold,
            "recovery_timeout": breaker.recovery_timeout
        }
    return status

def renew_tor_circuit():
    """Renew Tor circuit to get a fresh route."""
    try:
        from stem import Signal
        from stem.control import Controller
        
        with Controller.from_port(port=9051) as controller:
            controller.authenticate()
            controller.signal(Signal.NEWNYM)
            logger.info("Tor circuit renewed successfully")
    except Exception as e:
        logger.warning(f"Failed to renew Tor circuit: {e}")
        # Fallback: just wait a bit
        time.sleep(10)

def get_url_stats_db() -> dict:
    """Get statistics about URLs in database (thread-safe)."""
    with db_lock:
        conn = sqlite3.connect(VISITED_DB)
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM visited_urls WHERE visited=1')
        visited_count = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM visited_urls WHERE failed=1')
        failed_count = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM visited_urls WHERE visited=0 AND failed=0')
        pending_count = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM visited_urls')
        total_count = c.fetchone()[0]
        
        conn.close()
        
        return {
            'visited': visited_count,
            'failed': failed_count,
            'pending': pending_count,
            'total': total_count
        }

def auto_fill_links(query: str, limit: int = 20) -> dict:
    """
    Auto-fill function that searches for links and adds them directly to the database.
    This is a lightweight alternative to running the full agent process.
    
    Args:
        query: Search query to find links for
        limit: Maximum number of links to add (default 20)
    
    Returns:
        dict with links_added count and the actual links
    """
    try:
        logger.info(f"Auto-filling links for query: {query}")
        
        # Search for links using existing tor_search function
        links = tor_search(query, limit=limit)
        
        if not links:
            logger.warning(f"No links found for query: {query}")
            return {"links_added": 0, "links": []}
        
        # Add links to database as unvisited
        add_links_to_db_threadsafe(links)
        
        logger.info(f"Auto-fill completed: added {len(links)} links for query: {query}")
        
        return {
            "links_added": len(links),
            "links": links,
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Auto-fill failed for query {query}: {e}")
        return {"links_added": 0, "links": [], "error": str(e)}

# Example usage:
if __name__ == "__main__":
    # Run perpetual agent with 2-minute delays and 15 steps per run
    run_perpetual_agent(delay_seconds=10, max_steps_per_run=15)