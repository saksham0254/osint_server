# OSINT Dark Web Scraper API

A comprehensive FastAPI-based REST API for OSINT intelligence collection and search from dark web sources.

## 🚀 Features

- **Authentication & Session Management**
- **Intelligent Search** with semantic similarity and category filtering
- **Auto-Fill** - Automatically adds links to database when results are insufficient
- **Perpetual Agent** - Continuous background intelligence collection (auto-starts on server startup)
- **Real-time Statistics** - Category and URL statistics
- **Circuit Breaker Status** - Monitor search engine health
- **FAISS Vector Search** - High-performance semantic search across indexed documents

## 🏃‍♂️ Quick Start

### 1. Start the Server
```bash
python api.py
```
Server runs on: `http://localhost:8000`

**Note:** The perpetual agent automatically starts when the server starts with default settings:
- 2-minute delays between runs
- 15 steps per run
- Dynamic keyword generation enabled

### 2. Access API Documentation
Visit: `http://localhost:8000/docs`

### 3. Authentication
```bash
# Login
curl -X POST "http://localhost:8000/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'

# Response: {"success": true, "token": "your_token_here"}
```

## 📋 API Endpoints

### Authentication

#### POST `/login`
Login and get session token.
```json
{
  "username": "admin",
  "password": "admin"
}
```

#### POST `/signup`
Create new user account.
```json
{
  "username": "newuser",
  "password": "password123"
}
```

#### POST `/logout`
Logout and destroy session.

### 🔍 Intelligence Search

#### POST `/search`
Main search endpoint with auto-fill capability.
```json
{
  "query": "cyber threats",
  "top_k": 20,
  "sim_threshold": 0.2,
  "auto_collect": true
}
```

**Response:**
```json
{
  "success": true,
  "query": "cyber threats",
  "results": [
    {
      "similarity_score": 0.85,
      "url": "http://example.onion/",
      "title": "Cyber Threats Report",
      "chunk_id": 1,
      "text_content": "Detailed threat analysis...",
      "timestamp": "2024-01-01T00:00:00",
      "salient_phrases": ["malware", "ransomware"],
      "category": ["cyber", "malware"],
      "confidence": 85,
      "region": "US",
      "rationale": "Contains actionable cyber threat intelligence"
    }
  ],
  "count": 5,
  "auto_collection_started": true,
  "message": "Found 5 results. Started auto-fill for more links.",
  "parameters": {
    "top_k": 20,
    "sim_threshold": 0.2,
    "auto_collect": true
  }
}
```

**Auto-Fill Feature:**
- Automatically adds up to 20 links to database when results < 5
- Uses tor_search to find relevant .onion links
- Links are added as unvisited for future processing by the perpetual agent
- Non-blocking - returns immediate results
- Configurable with `auto_collect` parameter

### 🤖 Agent Control

#### POST `/agent/run`
Run agent for specified number of steps.
```json
{
  "n_steps": 10,
  "seed_keyword": "security threats"
}
```

#### POST `/agent/start-perpetual`
Start perpetual agent for continuous collection (if not already running).
```json
{
  "delay_seconds": 120,
  "max_steps_per_run": 15,
  "dynamic_keywords": true
}
```

**Note:** The perpetual agent automatically starts when the server starts. If you try to start it manually while it's already running, you'll get an error message.

#### POST `/agent/stop-perpetual`
Stop the perpetual agent.

#### GET `/agent/status`
Get perpetual agent status.
```json
{
  "success": true,
  "agent_status": {
    "running": true,
    "total_runs": 5,
    "last_run": "2024-01-01T12:00:00",
    "current_keyword": "cyber threats",
    "delay_seconds": 120,
    "max_steps_per_run": 15,
    "dynamic_keywords": true
  }
}
```

### 📊 Statistics

#### GET `/stats/categories`
Get intelligence statistics by category.
```json
{
  "success": true,
  "category_counts": {
    "cyber": 15,
    "malware": 8,
    "terrorism": 3,
    "fraud": 12
  }
}
```

#### GET `/stats/urls`
Get URL processing statistics.
```json
{
  "success": true,
  "url_stats": {
    "visited": 120,
    "failed": 15,
    "pending": 15,
    "total": 150
  }
}
```

#### GET `/latest-intelligence`
Get the latest 10 intelligence links with confidence > 50, ordered by timestamp.
```json
{
  "success": true,
  "links": [
    {
      "url": "http://example.onion/",
      "timestamp": "2024-01-01T12:00:00",
      "intelligence_relevant": true,
      "confidence": 85,
      "category": ["cyber", "malware"],
      "salient_phrases": ["malware", "ransomware"],
      "region": "US",
      "rationale": "Contains actionable cyber threat intelligence"
    }
  ]
}
```

### 🔧 System Status

#### GET `/status/circuit-breakers`
Get search engine circuit breaker status.
```json
{
  "success": true,
  "circuit_breakers": {
    "Ahmia": {"state": "CLOSED", "failure_count": 0},
    "Torch": {"state": "OPEN", "failure_count": 3},
    "Haystak": {"state": "CLOSED", "failure_count": 1}
  }
}
```

## 🔧 Configuration

### Perpetual Agent Auto-Start Settings
The perpetual agent automatically starts with these default settings:
- **Delay between runs**: 120 seconds (2 minutes)
- **Steps per run**: 15
- **Dynamic keywords**: Enabled
- **Search engines**: 5 engines with circuit breaker protection

### Auto-Fill Settings
- **Trigger threshold**: Results < 5
- **Links added**: Up to 20 per trigger
- **Background processing**: Non-blocking

## 🚨 Error Handling

All endpoints return consistent error format:
```json
{
  "success": false,
  "error": "Error description"
}
```

Common HTTP status codes:
- `200` - Success
- `400` - Bad request (missing parameters)
- `401` - Unauthorized (invalid/missing token)
- `500` - Internal server error

## 🔄 Auto-Reload

The server supports auto-reload for development:
- Automatically restarts when code changes
- Watches all files in project directory
- Hot reloads without manual restart

## 📝 Example Usage

### Complete Workflow
```bash
# 1. Login
TOKEN=$(curl -s -X POST "http://localhost:8000/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}' | jq -r '.token')

# 2. Search for intelligence (auto-fill will trigger if results < 5)
curl -H "Authorization: $TOKEN" \
  -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "cyber threats", "auto_collect": true}'

# 3. Check agent status (should be running from startup)
curl -H "Authorization: $TOKEN" \
  -X GET "http://localhost:8000/agent/status"

# 4. Get statistics
curl -H "Authorization: $TOKEN" \
  -X GET "http://localhost:8000/stats/categories"

# 5. Get latest intelligence
curl -H "Authorization: $TOKEN" \
  -X GET "http://localhost:8000/latest-intelligence"
```

## 🎯 Advanced Features

### Perpetual Agent Auto-Start
- Automatically starts when server starts
- Continuous background intelligence collection
- Dynamic keyword generation based on data gaps
- Configurable delays and step limits
- Graceful start/stop control

### Auto-Fill System
- Lightweight alternative to full agent processing
- Adds links to database for future processing
- Uses multiple search engines with redundancy
- Circuit breaker protection for fault tolerance

### Circuit Breakers
- Fault tolerance for search engines
- Automatic recovery after failures
- Status monitoring for all engines
- Prevents cascading failures

### FAISS Integration
- High-performance vector search
- 1536-dimensional embeddings
- Subset search for targeted queries
- Efficient similarity scoring

This API provides a complete OSINT intelligence collection and search platform with advanced automation capabilities. 