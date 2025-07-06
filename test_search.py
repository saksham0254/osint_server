#!/usr/bin/env python3
"""
Test script for the improved dark web search functionality
Tests URL validation, search engines, circuit breakers, and auto-fill features
"""

import agent
import time
import sys

def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("\nTesting circuit breaker...")
    
    # Test basic circuit breaker
    breaker = agent.CircuitBreaker(failure_threshold=2, recovery_timeout=5)
    
    # Test successful calls
    def success_func():
        return "success"
    
    for i in range(3):
        result = breaker.call(success_func)
        print(f"  Success call {i+1}: {result}")
    
    # Test failure handling
    def failure_func():
        raise Exception("Test failure")
    
    for i in range(3):
        try:
            result = breaker.call(failure_func)
            print(f"  Failure call {i+1}: {result}")
        except Exception as e:
            print(f"  Failure call {i+1}: {e}")
    
    # Test recovery
    print("  Waiting for circuit breaker to recover...")
    time.sleep(6)
    
    try:
        result = breaker.call(success_func)
        print(f"  Recovery test: {result}")
    except Exception as e:
        print(f"  Recovery test failed: {e}")

def test_url_validation():
    """Test URL validation functionality."""
    print("\nTesting URL validation...")
    
    test_urls = [
        "http://zqktlwi4fecvo6ri.onion/wiki/index.php/Main_Page",  # Valid
        "https://ahmia.fi/search/?q=test",  # Invalid (not .onion)
        "http://invalid.onion",  # Invalid (too short)
        "http://zqktlwi4fecvo6ri.onion/",  # Valid
        "ftp://zqktlwi4fecvo6ri.onion/",  # Invalid (wrong protocol)
        "http://zqktlwi4fecvo6ri.onion",  # Valid
    ]
    
    for url in test_urls:
        is_valid = agent.validate_onion_url(url)
        print(f"  {url}: {'✓' if is_valid else '✗'}")
    
    # Test filtering
    valid_urls = agent.filter_valid_onion_urls(test_urls)
    print(f"\nFiltered valid URLs: {valid_urls}")

def test_search_functionality():
    """Test the main search functionality."""
    print("\nTesting search functionality...")
    
    test_queries = [
        "cyber security",
        "privacy tools",
        "anonymous communication"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        try:
            links = agent.tor_search(query, limit=5)
            print(f"  Found {len(links)} links")
            for link in links:
                print(f"    - {link}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # Small delay between searches
        time.sleep(2)

def test_auto_fill_functionality():
    """Test the new auto-fill functionality."""
    print("\nTesting auto-fill functionality...")
    
    test_queries = [
        "weapons",
        "drugs",
        "cybersecurity"
    ]
    
    for query in test_queries:
        print(f"\nAuto-filling for: '{query}'")
        try:
            result = agent.auto_fill_links(query, limit=10)
            links_added = result.get('links_added', 0)
            links = result.get('links', [])
            
            print(f"  Added {links_added} links to database")
            if links:
                print(f"  Sample links:")
                for link in links[:3]:  # Show first 3
                    print(f"    - {link}")
                if len(links) > 3:
                    print(f"    ... and {len(links) - 3} more")
            
            if 'error' in result:
                print(f"  Error: {result['error']}")
                
        except Exception as e:
            print(f"  Error: {e}")
        
        # Small delay between tests
        time.sleep(2)

def test_faiss_search():
    """Test FAISS search functionality."""
    print("\nTesting FAISS search...")
    
    test_queries = [
        "cyber threats",
        "malware",
        "security vulnerabilities"
    ]
    
    for query in test_queries:
        print(f"\nSearching FAISS for: '{query}'")
        try:
            results = agent.search_by_category_and_semantics(query, top_k=5, sim_threshold=0.2)
            print(f"  Found {len(results)} results")
            
            for i, result in enumerate(results[:3]):  # Show first 3
                score = result.get('similarity_score', 0)
                url = result.get('url', 'N/A')
                title = result.get('title', 'N/A')
                confidence = result.get('confidence', 0)
                
                print(f"    {i+1}. {title}")
                print(f"       URL: {url}")
                print(f"       Score: {score:.3f}, Confidence: {confidence}%")
                
        except Exception as e:
            print(f"  Error: {e}")

def test_category_extraction():
    """Test category extraction from queries."""
    print("\nTesting category extraction...")
    
    test_queries = [
        "cyber attacks and malware",
        "terrorism threats in europe",
        "drug trafficking networks",
        "financial fraud schemes"
    ]
    
    for query in test_queries:
        print(f"\nExtracting categories from: '{query}'")
        try:
            categories = agent.extract_category_from_query(query)
            print(f"  Categories: {categories}")
        except Exception as e:
            print(f"  Error: {e}")

def test_database_functions():
    """Test database-related functions."""
    print("\nTesting database functions...")
    
    try:
        # Test category counts
        category_counts = agent.get_category_counts()
        print(f"  Category counts: {len(category_counts)} categories found")
        if category_counts:
            top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            for category, count in top_categories:
                print(f"    {category}: {count}")
        
        # Test URL stats
        url_stats = agent.get_url_stats_db()
        print(f"  URL stats: {url_stats}")
        
        # Test latest intelligence
        latest = agent.retrieve_condensed_pages(intelligence_relevant=1, confidence=50)
        print(f"  Latest intelligence: {len(latest)} items found")
        
    except Exception as e:
        print(f"  Error: {e}")

def main():
    """Run all tests."""
    print("=== Testing Improved Dark Web Search Functionality ===\n")
    
    # Test circuit breaker
    test_circuit_breaker()
    
    # Test URL validation
    test_url_validation()
    
    # Test category extraction
    test_category_extraction()
    
    # Test database functions
    test_database_functions()
    
    # Test FAISS search
    test_faiss_search()
    
    # Test auto-fill functionality
    test_auto_fill_functionality()
    
    # Test search functionality (this will actually make requests)
    print("\n" + "="*50)
    print("WARNING: The following test will make actual network requests.")
    print("Make sure Tor is running and you have internet connectivity.")
    print("="*50)
    
    response = input("\nContinue with network tests? (y/N): ")
    if response.lower() == 'y':
        test_search_functionality()
    else:
        print("Skipping network tests.")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    main() 