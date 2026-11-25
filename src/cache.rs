//! Expert caching system for MoE models with memory management

use std::collections::HashMap;
use std::collections::VecDeque;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_entries: usize,
}

/// LRU (Least Recently Used) cache implementation for MoE experts
pub struct ExpertCache {
    /// Cache storage with LRU tracking
    cache: HashMap<String, CachedExpert>,
    /// LRU order tracking using a deque for efficient operations
    lru_order: VecDeque<String>,
    /// Maximum cache size in bytes
    max_size: usize,
    /// Current cache size in bytes
    current_size: usize,
    /// Statistics tracking
    stats: CacheStats,
}

#[derive(Debug, Clone)]
pub struct CachedExpert {
    pub data: Vec<u8>,
    pub size: usize,
    pub last_accessed: Instant,
}

impl ExpertCache {
    /// Create a new expert cache with specified maximum size
    pub fn new(max_size: usize) -> Self {
        ExpertCache {
            cache: HashMap::new(),
            lru_order: VecDeque::new(),
            max_size,
            current_size: 0,
            stats: CacheStats {
                hits: 0,
                misses: 0,
                evictions: 0,
                total_entries: 0,
            },
        }
    }

    /// Insert an expert into the cache
    pub fn insert(&mut self, key: String, expert: CachedExpert) {
        // Remove if already exists
        if self.cache.contains_key(&key) {
            self.remove(&key);
        }

        // Evict if needed
        while self.current_size + expert.size > self.max_size && !self.lru_order.is_empty() {
            let evicted_key = self.lru_order.pop_front().unwrap();
            self.remove(&evicted_key);
        }

        // Add to cache
        self.cache.insert(key.clone(), expert);
        self.lru_order.push_back(key.clone());
        self.stats.total_entries = self.cache.len();
    }

    /// Get an expert from the cache
    pub fn get(&mut self, key: &str) -> Option<&CachedExpert> {
        // Check if key exists and get it
        if let Some(expert) = self.cache.get_mut(key) {
            // Update LRU tracking
            self.lru_order.retain(|k| k != key);
            self.lru_order.push_back(key.to_string());

            expert.last_accessed = Instant::now();
            self.stats.hits += 1;
            return Some(expert);
        }

        self.stats.misses += 1;
        None
    }

    /// Remove an expert from the cache
    fn remove(&mut self, key: &str) {
        if let Some(expert) = self.cache.remove(key) {
            self.current_size -= expert.size;
            self.lru_order.retain(|k| k != key);
            self.stats.evictions += 1;
            self.stats.total_entries = self.cache.len();
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru_order.clear();
        self.current_size = 0;
        self.stats = CacheStats {
            hits: 0,
            misses: 0,
            evictions: 0,
            total_entries: 0,
        };
    }
}