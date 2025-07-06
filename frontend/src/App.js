import React, { useState } from 'react';

const backgroundStyle = {
  minHeight: '100vh',
  background: 'linear-gradient(135deg, #232526 0%, #414345 100%)',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'flex-start',
  fontFamily: 'Roboto, Arial, sans-serif',
};

const cardStyle = {
  background: 'rgba(255,255,255,0.95)',
  boxShadow: '0 8px 32px 0 rgba(31,38,135,0.15)',
  borderRadius: 16,
  padding: 32,
  marginTop: 40,
  width: 600,
  maxWidth: '90vw',
};

const inputStyle = {
  width: '100%',
  padding: 12,
  borderRadius: 8,
  border: '1px solid #ccc',
  marginBottom: 16,
  fontSize: 16,
  background: '#f7f7f7',
};

const buttonStyle = {
  width: '100%',
  padding: 12,
  borderRadius: 8,
  border: 'none',
  background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
  color: '#fff',
  fontWeight: 600,
  fontSize: 16,
  cursor: 'pointer',
  marginTop: 8,
  boxShadow: '0 2px 8px 0 rgba(102,126,234,0.10)',
  transition: 'background 0.2s',
};

const errorStyle = {
  color: '#e74c3c',
  marginBottom: 12,
  textAlign: 'center',
};

const successStyle = {
  color: '#27ae60',
  marginBottom: 12,
  textAlign: 'center',
};

const infoStyle = {
  color: '#3498db',
  marginBottom: 12,
  textAlign: 'center',
  fontSize: 14,
};

const headerBar = {
  width: '100vw',
  background: 'rgba(34, 40, 49, 0.95)',
  color: '#fff',
  padding: '18px 0',
  textAlign: 'center',
  fontSize: 28,
  fontWeight: 700,
  letterSpacing: 1,
  boxShadow: '0 2px 8px 0 rgba(34,40,49,0.10)',
  marginBottom: 0,
};

const resultCard = {
  ...cardStyle,
  marginTop: 24,
  background: 'rgba(255,255,255,0.98)',
};

const phraseCard = {
  background: '#f8f9fa',
  border: '1px solid #e9ecef',
  borderRadius: 8,
  padding: 12,
  marginBottom: 8,
  fontSize: 14,
  color: '#495057',
};

const categoryTag = {
  display: 'inline-block',
  background: '#667eea',
  color: 'white',
  padding: '4px 8px',
  borderRadius: 12,
  fontSize: 12,
  marginRight: 6,
  marginBottom: 6,
};

// New styles for dashboard components
const dashboardContainer = {
  display: 'flex',
  gap: 24,
  marginTop: 24,
  flexWrap: 'wrap',
  justifyContent: 'center',
};

const dashboardCard = {
  ...cardStyle,
  marginTop: 0,
  flex: '1 1 400px',
  minWidth: 400,
  maxWidth: 500,
};

const categoryBubble = {
  display: 'inline-block',
  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
  color: 'white',
  padding: '12px 16px',
  borderRadius: '50px',
  fontSize: 14,
  margin: '8px',
  fontWeight: 600,
  boxShadow: '0 4px 12px rgba(102,126,234,0.3)',
  transition: 'transform 0.2s, box-shadow 0.2s',
  cursor: 'pointer',
};

const categoryBubbleHover = {
  transform: 'scale(1.05)',
  boxShadow: '0 6px 16px rgba(102,126,234,0.4)',
};

const linkFeedItem = {
  padding: 16,
  border: '1px solid #e9ecef',
  borderRadius: 8,
  marginBottom: 12,
  background: '#fafbfc',
  transition: 'box-shadow 0.2s',
};

const linkFeedItemHover = {
  boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
};

const confidenceBadge = {
  display: 'inline-block',
  padding: '4px 8px',
  borderRadius: 12,
  fontSize: 12,
  fontWeight: 600,
  marginLeft: 8,
};

// Navigation button styles
const navButtonStyle = {
  background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
  color: '#fff',
  border: 'none',
  borderRadius: 8,
  padding: '10px 20px',
  fontWeight: 600,
  fontSize: 14,
  cursor: 'pointer',
  marginRight: 12,
  boxShadow: '0 2px 8px 0 rgba(102,126,234,0.10)',
  transition: 'background 0.2s',
};

const navButtonDisabledStyle = {
  ...navButtonStyle,
  background: '#ccc',
  cursor: 'not-allowed',
  boxShadow: 'none',
};

const navigationBarStyle = {
  position: 'absolute',
  top: 24,
  left: 32,
  display: 'flex',
  gap: 12,
  zIndex: 10,
};

function App() {
  const [loggedIn, setLoggedIn] = useState(false);
  const [loginError, setLoginError] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showSignup, setShowSignup] = useState(false);
  const [signupError, setSignupError] = useState('');
  const [signupSuccess, setSignupSuccess] = useState('');
  const [signupUsername, setSignupUsername] = useState('');
  const [signupPassword, setSignupPassword] = useState('');

  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [searchMessage, setSearchMessage] = useState('');
  const [autoCollectionStarted, setAutoCollectionStarted] = useState(false);

  // New state for categories and latest links
  const [categories, setCategories] = useState({});
  const [latestLinks, setLatestLinks] = useState([]);
  const [loadingData, setLoadingData] = useState(false);

  // New state for navigation
  const [currentView, setCurrentView] = useState('dashboard'); // 'dashboard' or 'search'
  const [viewHistory, setViewHistory] = useState([]);

  React.useEffect(() => {
    // Add Google Font
    const link = document.createElement('link');
    link.href = 'https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap';
    link.rel = 'stylesheet';
    document.head.appendChild(link);
    return () => { document.head.removeChild(link); };
  }, []);

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoginError('');
    try {
      const res = await fetch('http://localhost:8000/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      const data = await res.json();
      if (data.success) {
        setLoggedIn(true);
        localStorage.setItem('token', data.token);
        // Clear any previous search states for clean user experience
        setSearchResults([]);
        setSearchQuery('');
        setSearchError('');
        setSearchMessage('');
        setAutoCollectionStarted(false);
        setSearchLoading(false);
        // Load dashboard data
        await loadDashboardData();
        // Reset navigation state
        setCurrentView('dashboard');
        setViewHistory([]);
      } else {
        setLoginError(data.error || 'Login failed');
      }
    } catch (err) {
      setLoginError('Network error');
    }
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    setSignupError('');
    setSignupSuccess('');
    try {
      const res = await fetch('http://localhost:8000/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: signupUsername, password: signupPassword })
      });
      const data = await res.json();
      if (data.success) {
        setSignupSuccess('Signup successful! You can now log in.');
        setShowSignup(false);
        setSignupUsername('');
        setSignupPassword('');
      } else {
        setSignupError(data.error || 'Signup failed');
      }
    } catch (err) {
      setSignupError('Network error');
    }
  };

  const handleLogout = async () => {
    try {
      await fetch('http://localhost:8000/logout', {
        method: 'POST',
        headers: { 'Authorization': localStorage.getItem('token') },
      });
    } catch (err) {
      // Ignore logout errors
    }
    setLoggedIn(false);
    setUsername('');
    setPassword('');
    // Clear all search-related states
    setSearchResults([]);
    setSearchQuery('');
    setSearchError('');
    setSearchMessage('');
    setAutoCollectionStarted(false);
    setSearchLoading(false);
    localStorage.removeItem('token');
    // Reset navigation state
    setCurrentView('dashboard');
    setViewHistory([]);
    setCategories({});
    setLatestLinks([]);
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    setSearchError('');
    setSearchMessage('');
    setSearchResults([]);
    setAutoCollectionStarted(false);
    setSearchLoading(true);
    
    // Navigate to search view
    navigateToView('search');
    
    try {
      const res = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': localStorage.getItem('token')
        },
        body: JSON.stringify({ 
          query: searchQuery,
          top_k: 20,
          sim_threshold: 0.2,
          auto_collect: true
        })
      });
      const data = await res.json();
      
      if (data.success) {
        setSearchResults(data.results || []);
        setSearchMessage(data.message || '');
        setAutoCollectionStarted(data.auto_collection_started || false);
      } else {
        setSearchError(data.error || 'Search failed');
      }
    } catch (err) {
      setSearchError('Network error');
    }
    setSearchLoading(false);
  };

  const logoutButtonStyle = {
    position: 'absolute',
    top: 24,
    right: 32,
    background: 'linear-gradient(90deg, #e74c3c 0%, #ff7675 100%)',
    color: '#fff',
    border: 'none',
    borderRadius: 8,
    padding: '10px 24px',
    fontWeight: 600,
    fontSize: 16,
    cursor: 'pointer',
    boxShadow: '0 2px 8px 0 rgba(231,76,60,0.10)',
    zIndex: 10
  };

  const modalOverlayStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100vw',
    height: '100vh',
    background: 'rgba(0,0,0,0.35)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000
  };

  const modalStyle = {
    ...cardStyle,
    width: 380,
    maxWidth: '95vw',
    position: 'relative',
    boxShadow: '0 12px 32px 0 rgba(31,38,135,0.18)',
    padding: 36,
    marginTop: 0
  };

  const renderSearchResults = () => {
    if (searchResults.length === 0) {
      return (
        <div style={resultCard}>
          <h3 style={{ color: '#232526', marginBottom: 12, textAlign: 'center' }}>No Results Found</h3>
          {autoCollectionStarted ? (
            <div style={infoStyle}>
              <p>🔍 Auto-fill has been triggered with your query.</p>
              <p>Links are being added to the database for future processing.</p>
              <p><strong>Query:</strong> {searchQuery}</p>
            </div>
          ) : (
            <div style={infoStyle}>
              <p>No intelligence found for your query.</p>
              <p>Try different keywords or check back later.</p>
            </div>
          )}
        </div>
      );
    }

    return (
      <div style={resultCard}>
        <h3 style={{ color: '#232526', marginBottom: 16 }}>
          Search Results ({searchResults.length})
          {autoCollectionStarted && (
            <span style={{ fontSize: 14, color: '#3498db', marginLeft: 12 }}>
              🔄 Auto-fill active
            </span>
          )}
        </h3>
        {searchMessage && (
          <div style={infoStyle}>{searchMessage}</div>
        )}
        {searchResults.map((result, i) => (
          <div key={i} style={{ 
            marginBottom: 24, 
            padding: 16, 
            border: '1px solid #e9ecef', 
            borderRadius: 8,
            background: '#fafbfc'
          }}>
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontWeight: 600, marginBottom: 6, fontSize: 16 }}>
                <a href={result.url} target="_blank" rel="noopener noreferrer" 
                   style={{ color: '#764ba2', textDecoration: 'underline' }}>
                  {result.title || result.url}
                </a>
              </div>
              <div style={{ fontSize: 14, color: '#6c757d', marginBottom: 8 }}>
                Score: {(result.similarity_score * 100).toFixed(1)}% | 
                Confidence: {result.confidence}% | 
                Region: {result.region || 'Unknown'}
              </div>
            </div>
            
            {result.category && result.category.length > 0 && (
              <div style={{ marginBottom: 12 }}>
                {result.category.map((cat, idx) => (
                  <span key={idx} style={categoryTag}>{cat}</span>
                ))}
              </div>
            )}
            
            {result.rationale && (
              <div style={{ 
                fontSize: 13, 
                color: '#6c757d', 
                fontStyle: 'italic',
                background: '#f8f9fa',
                padding: 8,
                borderRadius: 4,
                borderLeft: '3px solid #667eea'
              }}>
                <strong>Analysis:</strong> {result.rationale}
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  const fetchCategories = async () => {
    try {
      const res = await fetch('http://localhost:8000/stats/categories', {
        headers: { 'Authorization': localStorage.getItem('token') }
      });
      const data = await res.json();
      if (data.success) {
        setCategories(data.category_counts || {});
      }
    } catch (err) {
      console.error('Failed to fetch categories:', err);
    }
  };

  const fetchLatestLinks = async () => {
    try {
      const res = await fetch('http://localhost:8000/latest-intelligence', {
        headers: { 'Authorization': localStorage.getItem('token') }
      });
      const data = await res.json();
      if (data.success) {
        setLatestLinks(data.links || []);
      }
    } catch (err) {
      console.error('Failed to fetch latest links:', err);
    }
  };

  const loadDashboardData = async () => {
    setLoadingData(true);
    await Promise.all([fetchCategories(), fetchLatestLinks()]);
    setLoadingData(false);
  };

  const handleBack = () => {
    if (viewHistory.length > 0) {
      const previousView = viewHistory[viewHistory.length - 1];
      setViewHistory(viewHistory.slice(0, -1));
      setCurrentView(previousView);
      
      if (previousView === 'dashboard') {
        // Clear search results when going back to dashboard
        setSearchResults([]);
        setSearchMessage('');
        setAutoCollectionStarted(false);
        setSearchError('');
      }
    }
  };

  const handleRefresh = async () => {
    if (currentView === 'dashboard') {
      await loadDashboardData();
    } else if (currentView === 'search' && searchQuery) {
      // Refresh search results
      setSearchLoading(true);
      setSearchError('');
      setSearchMessage('');
      setSearchResults([]);
      setAutoCollectionStarted(false);
      
      try {
        const res = await fetch('http://localhost:8000/search', {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Authorization': localStorage.getItem('token')
          },
          body: JSON.stringify({ 
            query: searchQuery,
            top_k: 20,
            sim_threshold: 0.2,
            auto_collect: true
          })
        });
        const data = await res.json();
        
        if (data.success) {
          setSearchResults(data.results || []);
          setSearchMessage(data.message || '');
          setAutoCollectionStarted(data.auto_collection_started || false);
        } else {
          setSearchError(data.error || 'Search failed');
        }
      } catch (err) {
        setSearchError('Network error');
      }
      setSearchLoading(false);
    }
  };

  const navigateToView = (view) => {
    setViewHistory([...viewHistory, currentView]);
    setCurrentView(view);
  };

  const renderCategoryBubbles = () => {
    if (Object.keys(categories).length === 0) {
      return (
        <div style={dashboardCard}>
          <h3 style={{ color: '#232526', marginBottom: 16, textAlign: 'center' }}>
            🏷️ Intelligence Categories
          </h3>
          <div style={infoStyle}>
            {loadingData ? 'Loading categories...' : 'No categories found'}
          </div>
        </div>
      );
    }

    return (
      <div style={dashboardCard}>
        <h3 style={{ color: '#232526', marginBottom: 16, textAlign: 'center' }}>
          🏷️ Intelligence Categories
        </h3>
        <div style={{ textAlign: 'center' }}>
          {Object.entries(categories).map(([category, count]) => (
            <div
              key={category}
              style={categoryBubble}
              onMouseEnter={(e) => {
                e.target.style.transform = categoryBubbleHover.transform;
                e.target.style.boxShadow = categoryBubbleHover.boxShadow;
              }}
              onMouseLeave={(e) => {
                e.target.style.transform = '';
                e.target.style.boxShadow = '';
              }}
              onClick={async () => {
                setSearchQuery(category);
                setSearchError('');
                setSearchMessage('');
                setSearchResults([]);
                setAutoCollectionStarted(false);
                setSearchLoading(true);
                
                // Navigate to search view
                navigateToView('search');
                
                try {
                  const res = await fetch('http://localhost:8000/search', {
                    method: 'POST',
                    headers: { 
                      'Content-Type': 'application/json',
                      'Authorization': localStorage.getItem('token')
                    },
                    body: JSON.stringify({ 
                      query: category,
                      top_k: 20,
                      sim_threshold: 0.2,
                      auto_collect: true
                    })
                  });
                  const data = await res.json();
                  
                  if (data.success) {
                    setSearchResults(data.results || []);
                    setSearchMessage(data.message || '');
                    setAutoCollectionStarted(data.auto_collection_started || false);
                  } else {
                    setSearchError(data.error || 'Search failed');
                  }
                } catch (err) {
                  setSearchError('Network error');
                }
                setSearchLoading(false);
              }}
            >
              {category} ({count})
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderLatestLinksFeed = () => {
    if (latestLinks.length === 0) {
      return (
        <div style={dashboardCard}>
          <h3 style={{ color: '#232526', marginBottom: 16, textAlign: 'center' }}>
            📰 Latest Intelligence
          </h3>
          <div style={infoStyle}>
            {loadingData ? 'Loading latest intelligence...' : 'No recent intelligence found'}
          </div>
        </div>
      );
    }

    return (
      <div style={dashboardCard}>
        <h3 style={{ color: '#232526', marginBottom: 16, textAlign: 'center' }}>
          📰 Latest Intelligence
        </h3>
        <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
          {latestLinks.map((link, index) => {
            const confidenceColor = link.confidence >= 80 ? '#27ae60' : 
                                  link.confidence >= 60 ? '#f39c12' : '#e74c3c';
            
            return (
              <div
                key={index}
                style={linkFeedItem}
                onMouseEnter={(e) => {
                  e.target.style.boxShadow = linkFeedItemHover.boxShadow;
                }}
                onMouseLeave={(e) => {
                  e.target.style.boxShadow = '';
                }}
              >
                <div style={{ marginBottom: 8 }}>
                  <a 
                    href={link.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    style={{ 
                      color: '#764ba2', 
                      textDecoration: 'underline',
                      fontWeight: 600,
                      fontSize: 14
                    }}
                  >
                    {link.url}
                  </a>
                  <span 
                    style={{
                      ...confidenceBadge,
                      background: confidenceColor,
                      color: 'white'
                    }}
                  >
                    {link.confidence}%
                  </span>
                </div>
                
                {link.category && link.category.length > 0 && (
                  <div style={{ marginBottom: 8 }}>
                    {link.category.map((cat, idx) => (
                      <span key={idx} style={categoryTag}>{cat}</span>
                    ))}
                  </div>
                )}
                
                {link.salient_phrases && link.salient_phrases.length > 0 && (
                  <div style={{ fontSize: 13, color: '#6c757d', fontStyle: 'italic' }}>
                    "{link.salient_phrases[0]}"
                    {link.salient_phrases.length > 1 && ' ...'}
                  </div>
                )}
                
                <div style={{ fontSize: 12, color: '#6c757d', marginTop: 8 }}>
                  {new Date(link.timestamp).toLocaleString()}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div style={{ ...backgroundStyle, position: 'relative' }}>
      <div style={headerBar}>OSINT Intelligence Search</div>
      {loggedIn && (
        <>
          <button onClick={handleLogout} style={logoutButtonStyle}>Logout</button>
          
          {/* Navigation Bar */}
          <div style={navigationBarStyle}>
            <button 
              onClick={handleBack} 
              style={viewHistory.length > 0 ? navButtonStyle : navButtonDisabledStyle}
              disabled={viewHistory.length === 0}
            >
              ← Back
            </button>
            <button onClick={handleRefresh} style={navButtonStyle}>
              🔄 Refresh
            </button>
          </div>
        </>
      )}
      {!loggedIn && (
        <>
          <form onSubmit={handleLogin} style={cardStyle}>
            <h2 style={{ textAlign: 'center', marginBottom: 24, color: '#232526' }}>Login</h2>
            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={e => setUsername(e.target.value)}
              style={inputStyle}
              autoFocus
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              style={inputStyle}
            />
            {loginError && <div style={errorStyle}>{loginError}</div>}
            <button type="submit" style={buttonStyle}>Login</button>
            <button type="button" style={{ ...buttonStyle, background: '#eee', color: '#232526', marginTop: 8 }} onClick={() => { setShowSignup(true); setLoginError(''); }}>Sign Up</button>
          </form>
          {showSignup && (
            <div style={modalOverlayStyle}>
              <form onSubmit={handleSignup} style={modalStyle}>
                <h2 style={{ textAlign: 'center', marginBottom: 24, color: '#232526' }}>Sign Up</h2>
                <input
                  type="text"
                  placeholder="Username"
                  value={signupUsername}
                  onChange={e => setSignupUsername(e.target.value)}
                  style={inputStyle}
                  autoFocus
                />
                <input
                  type="password"
                  placeholder="Password"
                  value={signupPassword}
                  onChange={e => setSignupPassword(e.target.value)}
                  style={inputStyle}
                />
                {signupError && <div style={errorStyle}>{signupError}</div>}
                {signupSuccess && <div style={successStyle}>{signupSuccess}</div>}
                <button type="submit" style={buttonStyle}>Sign Up</button>
                <button type="button" style={{ ...buttonStyle, background: '#eee', color: '#232526', marginTop: 8 }} onClick={() => { setShowSignup(false); setSignupError(''); setSignupSuccess(''); }}>Cancel</button>
              </form>
            </div>
          )}
        </>
      )}
      {loggedIn && (
        <>
          {currentView === 'dashboard' && (
            <>
              <form onSubmit={handleSearch} style={cardStyle}>
                <h2 style={{ textAlign: 'center', marginBottom: 24, color: '#232526' }}>Intelligence Search</h2>
                <input
                  type="text"
                  placeholder="Search for cyber threats, malware, terrorism, fraud, etc..."
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  style={inputStyle}
                  required
                />
                {searchError && <div style={errorStyle}>{searchError}</div>}
                <button type="submit" style={buttonStyle} disabled={searchLoading}>
                  {searchLoading ? 'Searching...' : '🔍 Search Intelligence'}
                </button>
              </form>
              
              {/* Dashboard Components */}
              <div style={dashboardContainer}>
                {renderCategoryBubbles()}
                {renderLatestLinksFeed()}
              </div>
            </>
          )}
          
          {currentView === 'search' && (
            <>
              <form onSubmit={handleSearch} style={cardStyle}>
                <h2 style={{ textAlign: 'center', marginBottom: 24, color: '#232526' }}>Search Results</h2>
                <input
                  type="text"
                  placeholder="Search for cyber threats, malware, terrorism, fraud, etc..."
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  style={inputStyle}
                  required
                />
                {searchError && <div style={errorStyle}>{searchError}</div>}
                <button type="submit" style={buttonStyle} disabled={searchLoading}>
                  {searchLoading ? 'Searching...' : '🔍 Search Intelligence'}
                </button>
              </form>
              {(searchResults.length > 0 || searchMessage || autoCollectionStarted) && renderSearchResults()}
            </>
          )}
        </>
      )}
    </div>
  );
}

export default App;
