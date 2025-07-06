# OSINT Server

This project is an educational platform for open source intelligence (OSINT) research and experimentation. It includes a Python backend (with Tor proxy support) and a frontend for interacting with the system.

---

## Features
- Tor proxy integration for anonymous web requests
- Python backend for OSINT tasks
- (Optional) Frontend for user interaction

---

## Getting Started

### 1. Clone the Repository
```sh
git clone <repo_url>
cd osint_server
```

### 2. Create and Activate a Python Virtual Environment
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Start the Tor Proxy
- **Install Tor:**
  - On macOS: `brew install tor`
  - On Ubuntu: `sudo apt-get install tor`
- **Start Tor:**
```sh
tor &  # Starts Tor in the background
```
- By default, Tor listens on `localhost:9050` (SOCKS5).

### 5. Configure Environment Variables
Create a `.env` file in the project root (if needed):
```
# Example .env
TOR_SOCKS_PROXY=localhost:9050
# Add other environment variables as needed
```

### 6. Run the Python Backend
```sh
python api.py
```

### 7. (Optional) Run the Frontend
If you have a frontend (e.g., React):
```sh
cd frontend
npm install
npm start
```

---

## Usage Notes
- The backend will route requests through Tor for anonymity.
- Make sure Tor is running before starting the backend.
- Configure any API keys or secrets in your `.env` file.

---

## Troubleshooting
- If you have issues connecting through Tor, check that the Tor service is running and listening on the correct port.
- For Python package issues, try upgrading pip: `pip install --upgrade pip`

---

## License
This project is for educational purposes only.
