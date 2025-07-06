from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import getpass
import sqlite3
import hashlib
import threading
import logging
import asyncio
import agent  # Import our agent module for advanced functionality
from datetime import datetime
import signal
# Create a thread pool executor for blocking operations
import concurrent.futures
import atexit
import os
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Credentials (same as scraper.py)
APP_USERNAME = 'admin'
APP_PASSWORD = 'admin'

# Tor proxy
TOR_PROXY = {
    'http': 'socks5h://127.0.0.1:9050',
    'https': 'socks5h://127.0.0.1:9050'
}

# Seedlist of URLs to crawl
SEEDLIST = [
    "http://zqktlwi4fecvo6ri.onion/wiki/index.php/Main_Page",
    "http://duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad.onion/",
    # Add more .onion or clearnet URLs here
]

DB_PATH = 'users.db'

# Global variable to track the perpetual agent
perpetual_agent_instance = None
perpetual_agent_thread = None

# Create daemon thread pool executor to ensure threads don't prevent exit
thread_pool_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="osint_worker"
)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

init_db()

# Load FAISS index from database on startup
try:
    agent.load_faiss_index_from_db()
    logger.info("FAISS index loaded successfully from database")
except Exception as e:
    logger.error(f"Failed to load FAISS index: {e}")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- Pydantic models ---
class SignupRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class RagSearchRequest(BaseModel):
    query: str

# --- In-memory session (for demo, not production) ---
sessions = {}

def get_current_user(request: Request):
    session_token = request.headers.get('Authorization')
    if not session_token or session_token not in sessions:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return sessions[session_token]

import secrets

def create_session(username):
    token = secrets.token_hex(32)
    sessions[token] = username
    return token

def destroy_session(token):
    if token in sessions:
        del sessions[token]

# --- Endpoints ---
@app.post('/signup')
async def signup(data: SignupRequest):
    username = data.username
    password = data.password
    if not username or not password:
        return JSONResponse({'success': False, 'error': 'Username and password required'}, status_code=400)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hash_password(password)))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return JSONResponse({'success': False, 'error': 'Username already exists'}, status_code=400)
    conn.close()
    return {'success': True}

@app.post('/login')
async def login(data: LoginRequest):
    username = data.username
    password = data.password
    if not username or not password:
        return JSONResponse({'success': False, 'error': 'Username and password required'}, status_code=400)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    if row and row[0] == hash_password(password):
        token = create_session(username)
        return {'success': True, 'token': token}
    else:
        return JSONResponse({'success': False, 'error': 'Invalid credentials'}, status_code=401)

@app.post('/logout')
async def logout(request: Request):
    session_token = request.headers.get('Authorization')
    destroy_session(session_token)
    return {'success': True}

# --- New endpoints for improved functionality ---

@app.get('/status/circuit-breakers')
async def get_circuit_breaker_status(user=Depends(get_current_user)):
    """Get the status of all search engine circuit breakers."""
    try:
        status = await asyncio.get_event_loop().run_in_executor(
            thread_pool_executor, 
            agent.get_circuit_breaker_status
        )
        return {'success': True, 'circuit_breakers': status}
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

# Register cleanup function to run on exit
def cleanup_on_exit():
    """Cleanup when process exits."""
    try:
        logger.info("Cleanup on exit...")
        
        # Shutdown thread pool gracefully
        thread_pool_executor.shutdown(wait=False)
        
        # Log any remaining threads for debugging
        import threading
        remaining_threads = [t for t in threading.enumerate() if t != threading.main_thread()]
        if remaining_threads:
            logger.info(f"Remaining threads on exit: {[t.name for t in remaining_threads]}")
            
    except Exception as e:
        logger.error(f"Error in cleanup: {e}")

atexit.register(cleanup_on_exit)

# Note: Let uvicorn handle signals gracefully
# Custom signal handlers can interfere with uvicorn's shutdown process
# and cause issues with proper cleanup

@app.on_event("startup")
async def startup_event():
    """Start the perpetual agent when the application starts."""
    global perpetual_agent_instance, perpetual_agent_thread
    
    try:
        logger.info("Initializing application on startup...")
        
        # Initialize databases
        init_db()
        agent.init_db_tables()
        agent.init_condensed_db()
        
        # Load FAISS index from database
        agent.load_faiss_index_from_db()
        
        logger.info("Databases and FAISS index initialized successfully")
        
        # Create and start the perpetual agent with default settings
        perpetual_agent_instance = agent.PerpetualAgent(
            delay_seconds=120,  # 2 minutes between runs
            max_steps_per_run=15,  # 15 steps per run
            dynamic_keywords=True  # Use dynamic keyword generation
        )
        
        # Start the agent in a background thread
        perpetual_agent_thread = threading.Thread(target=perpetual_agent_instance.start, daemon=True)
        perpetual_agent_thread.start()
        
        logger.info("Perpetual agent started successfully on application startup")
        
    except Exception as e:
        logger.error(f"Failed to start perpetual agent on startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the perpetual agent when the application shuts down."""
    global perpetual_agent_instance, perpetual_agent_thread, thread_pool_executor
    
    logger.info("Starting shutdown...")
    
    # Quick shutdown - don't wait for anything
    try:
        if perpetual_agent_instance and perpetual_agent_instance.running:
            logger.info("Stopping perpetual agent...")
            perpetual_agent_instance.stop()
    except Exception as e:
        logger.error(f"Error stopping perpetual agent: {e}")
    
    try:
        logger.info("Shutting down thread pool...")
        thread_pool_executor.shutdown(wait=False)
        logger.info("Thread pool shutdown initiated")
    except Exception as e:
        logger.error(f"Error shutting down thread pool: {e}")
    
    logger.info("Shutdown completed")

@app.post('/agent/start-perpetual')
async def start_perpetual_agent(data: dict, user=Depends(get_current_user)):
    """Start the perpetual agent process."""
    global perpetual_agent_instance, perpetual_agent_thread
    
    try:
        delay_seconds = data.get('delay_seconds', 120)
        max_steps_per_run = data.get('max_steps_per_run', 15)
        dynamic_keywords = data.get('dynamic_keywords', True)
        
        # Check if agent is already running
        if perpetual_agent_instance and perpetual_agent_instance.running:
            return JSONResponse({
                'success': False, 
                'error': 'Perpetual agent is already running',
                'message': 'The agent was started automatically on application startup. Use /agent/stop-perpetual to stop it first.'
            }, status_code=400)
        
        # Create and start the perpetual agent
        perpetual_agent_instance = agent.PerpetualAgent(
            delay_seconds=delay_seconds,
            max_steps_per_run=max_steps_per_run,
            dynamic_keywords=dynamic_keywords
        )
        
        # Start the agent in a background thread
        perpetual_agent_thread = threading.Thread(target=perpetual_agent_instance.start, daemon=True)
        perpetual_agent_thread.start()
        
        return {
            'success': True, 
            'message': f'Perpetual agent started with {delay_seconds}s delay, {max_steps_per_run} steps per run',
            'parameters': {
                'delay_seconds': delay_seconds,
                'max_steps_per_run': max_steps_per_run,
                'dynamic_keywords': dynamic_keywords
            }
        }
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.post('/agent/stop-perpetual')
async def stop_perpetual_agent(user=Depends(get_current_user)):
    """Stop the perpetual agent process."""
    global perpetual_agent_instance
    
    try:
        if perpetual_agent_instance and perpetual_agent_instance.running:
            perpetual_agent_instance.stop()
            return {'success': True, 'message': 'Perpetual agent stopped'}
        else:
            return JSONResponse({'success': False, 'error': 'No perpetual agent is running'}, status_code=400)
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.get('/agent/status')
async def get_agent_status(user=Depends(get_current_user)):
    """Get the status of the perpetual agent."""
    global perpetual_agent_instance
    
    try:
        if perpetual_agent_instance:
            status = perpetual_agent_instance.get_status()
            return {'success': True, 'agent_status': status}
        else:
            return {'success': True, 'agent_status': {'running': False, 'message': 'No agent instance'}}
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.get('/stats/categories')
async def get_category_stats(user=Depends(get_current_user)):
    """Get statistics about collected intelligence by category."""
    try:
        counts = await asyncio.get_event_loop().run_in_executor(
            thread_pool_executor, 
            agent.get_category_counts
        )
        return {'success': True, 'category_counts': counts}
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.get('/stats/urls')
async def get_url_stats(user=Depends(get_current_user)):
    """Get statistics about URLs in the database."""
    try:
        stats = await asyncio.get_event_loop().run_in_executor(
            thread_pool_executor, 
            agent.get_url_stats_db
        )
        return {'success': True, 'url_stats': stats}
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.get('/health')
async def health_check():
    """Health check endpoint to monitor system status."""
    try:
        # Check thread pool status
        thread_pool_status = {
            "active_threads": thread_pool_executor._threads,
            "pending_tasks": thread_pool_executor._work_queue.qsize(),
            "max_workers": thread_pool_executor._max_workers
        }
        
        # Check perpetual agent status
        agent_status = None
        if perpetual_agent_instance:
            agent_status = perpetual_agent_instance.get_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "thread_pool": thread_pool_status,
            "perpetual_agent": agent_status
        }
    except Exception as e:
        return JSONResponse(
            {"status": "unhealthy", "error": str(e)}, 
            status_code=500
        )

@app.get('/latest-intelligence')
async def get_latest_intelligence(user=Depends(get_current_user)):
    """Get the latest 10 intelligence links with confidence > 50, ordered by timestamp."""
    try:
        # Get latest intelligence from the condensed pages table
        latest = await asyncio.get_event_loop().run_in_executor(
            thread_pool_executor, 
            agent.retrieve_condensed_pages, 
            None, 1, 50, None
        )
        
        latest.sort(
            key=lambda x: (
                1 if x.get('last_updated') else 0,
                x.get('last_updated') or x.get('timestamp', '')
            ),
            reverse=True
        )
        latest_10 = latest[:20]
        
        return {'success': True, 'links': latest_10}
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.post('/search')
async def search_endpoint(data: dict, user=Depends(get_current_user)):
    """Search collected intelligence by category and semantic similarity."""
    try:
        query = data.get('query', '')
        top_k = data.get('top_k', 20)
        sim_threshold = data.get('sim_threshold', 0.2)
        auto_collect = data.get('auto_collect', True)  # New parameter to control auto-collection
        
        if not query:
            return JSONResponse({'success': False, 'error': 'Query required'}, status_code=400)
        
        # Run the agent's search function in a thread pool to avoid blocking
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                thread_pool_executor, 
                agent.search_by_category_and_semantics, 
                query, top_k, sim_threshold
            )
        except asyncio.CancelledError:
            logger.warning("Search was cancelled during shutdown")
            return JSONResponse({'success': False, 'error': 'Operation cancelled during shutdown'}, status_code=503)
        
        # If results are insufficient and auto_collect is enabled, auto-fill links
        if len(results) < 5 and auto_collect:
            try:
                logger.info(f"Auto-filling links for query: {query}")
                # Run auto-fill in thread pool to avoid blocking
                auto_fill_result = await asyncio.get_event_loop().run_in_executor(
                    thread_pool_executor, 
                    agent.auto_fill_links, 
                    query, 20
                )
                logger.info(f"Auto-fill completed for query: {query} - added {auto_fill_result.get('links_added', 0)} links")
                
                return {
                    'success': True, 
                    'query': query,
                    'results': results, 
                    'count': len(results),
                    'auto_collection_started': True,
                    'auto_fill_result': auto_fill_result,
                    'message': f'Found {len(results)} results. Auto-fill completed and added {auto_fill_result.get("links_added", 0)} links.',
                    'parameters': {
                        'top_k': top_k,
                        'sim_threshold': sim_threshold,
                        'auto_collect': auto_collect
                    }
                }
            except Exception as e:
                logger.error(f"Auto-fill failed for query {query}: {e}")
                return {
                    'success': True, 
                    'query': query,
                    'results': results, 
                    'count': len(results),
                    'auto_collection_started': False,
                    'auto_fill_error': str(e),
                    'message': f'Found {len(results)} results. Auto-fill failed: {e}',
                    'parameters': {
                        'top_k': top_k,
                        'sim_threshold': sim_threshold,
                        'auto_collect': auto_collect
                    }
                }
        else:
            return {
                'success': True, 
                'query': query,
                'results': results, 
                'count': len(results),
                'auto_collection_started': False,
                'message': f'Found {len(results)} results.',
                'parameters': {
                    'top_k': top_k,
                    'sim_threshold': sim_threshold,
                    'auto_collect': auto_collect
                }
            }
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.post('/agent/run')
async def run_agent_endpoint(data: dict, user=Depends(get_current_user)):
    """Run the OSINT agent for a specified number of steps."""
    try:
        n_steps = data.get('n_steps', 10)
        seed_keyword = data.get('seed_keyword', 'security threats')
        
        # Run the agent in a thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(
            thread_pool_executor, 
            agent.run_agent, 
            n_steps, seed_keyword
        )
        return {'success': True, 'message': f'Agent completed {n_steps} steps'}
    except asyncio.CancelledError:
        logger.warning("Agent run was cancelled during shutdown")
        return JSONResponse({'success': False, 'error': 'Operation cancelled during shutdown'}, status_code=503)
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

if __name__ == '__main__':
    uvicorn.run(
        "api:app", 
        host='0.0.0.0', 
        port=8000,  # Use different port to avoid conflicts
        reload=False,  # reload=True with multiple workers causes conflicts
        workers=1,  # Start with 1 worker to eliminate worker conflicts
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10  # Shorter timeout for faster shutdown
    )