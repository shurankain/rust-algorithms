// LRU Cache - Least Recently Used Cache
//
// A cache that evicts the least recently used items when capacity is exceeded.
// Uses a HashMap for O(1) lookups and a doubly-linked list for O(1) eviction ordering.
//
// Time Complexity:
// - get: O(1)
// - put: O(1)
// - remove: O(1)
//
// Space Complexity: O(capacity)
//
// Common use cases:
// - Memory caching (web browsers, databases)
// - Page replacement algorithms
// - Memoization with bounded memory

use std::collections::HashMap;
use std::hash::Hash;
use std::ptr::NonNull;

// Node in the doubly-linked list
struct Node<K, V> {
    key: K,
    value: V,
    prev: Option<NonNull<Node<K, V>>>,
    next: Option<NonNull<Node<K, V>>>,
}

impl<K, V> Node<K, V> {
    fn new(key: K, value: V) -> Self {
        Self {
            key,
            value,
            prev: None,
            next: None,
        }
    }
}

/// LRU Cache with O(1) operations
pub struct LruCache<K, V> {
    capacity: usize,
    map: HashMap<K, NonNull<Node<K, V>>>,
    // Dummy head and tail for easier list manipulation
    head: Option<NonNull<Node<K, V>>>,
    tail: Option<NonNull<Node<K, V>>>,
}

impl<K, V> LruCache<K, V>
where
    K: Hash + Eq + Clone,
{
    /// Create a new LRU cache with the given capacity
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be positive");
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            head: None,
            tail: None,
        }
    }

    /// Get a value from the cache. Returns None if not present.
    /// Marks the item as recently used.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let node_ptr = *self.map.get(key)?;

        // Move to front (most recently used)
        self.move_to_front(node_ptr);

        // Safety: we just verified the node exists
        unsafe { Some(&(*node_ptr.as_ptr()).value) }
    }

    /// Get a mutable reference to a value. Returns None if not present.
    /// Marks the item as recently used.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let node_ptr = *self.map.get(key)?;

        self.move_to_front(node_ptr);

        // Safety: we just verified the node exists
        unsafe { Some(&mut (*node_ptr.as_ptr()).value) }
    }

    /// Insert or update a key-value pair.
    /// If key exists, updates the value and marks as recently used.
    /// If at capacity, evicts the least recently used item.
    /// Returns the old value if key existed.
    pub fn put(&mut self, key: K, value: V) -> Option<V> {
        if let Some(&node_ptr) = self.map.get(&key) {
            // Key exists, update value and move to front
            let old_value = unsafe {
                let node = node_ptr.as_ptr();
                std::mem::replace(&mut (*node).value, value)
            };
            self.move_to_front(node_ptr);
            return Some(old_value);
        }

        // Key doesn't exist, need to insert
        if self.map.len() >= self.capacity {
            // Evict least recently used (tail)
            self.evict_lru();
        }

        // Create new node
        let node = Box::new(Node::new(key.clone(), value));
        let node_ptr = NonNull::new(Box::into_raw(node)).unwrap();

        // Add to front
        self.push_front(node_ptr);
        self.map.insert(key, node_ptr);

        None
    }

    /// Remove a key from the cache. Returns the value if present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let node_ptr = self.map.remove(key)?;

        // Remove from linked list
        self.unlink(node_ptr);

        // Free the node and return value
        let node = unsafe { Box::from_raw(node_ptr.as_ptr()) };
        Some(node.value)
    }

    /// Check if key exists in cache (doesn't affect LRU order)
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Get the number of items in the cache
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Get the capacity of the cache
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all items from the cache
    pub fn clear(&mut self) {
        // Free all nodes
        let mut current = self.head;
        while let Some(node_ptr) = current {
            unsafe {
                current = (*node_ptr.as_ptr()).next;
                drop(Box::from_raw(node_ptr.as_ptr()));
            }
        }

        self.map.clear();
        self.head = None;
        self.tail = None;
    }

    /// Peek at a value without affecting LRU order
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map
            .get(key)
            .map(|&node_ptr| unsafe { &(*node_ptr.as_ptr()).value })
    }

    /// Get the least recently used key-value pair without removing it
    pub fn peek_lru(&self) -> Option<(&K, &V)> {
        self.tail.map(|node_ptr| unsafe {
            let node = node_ptr.as_ptr();
            (&(*node).key, &(*node).value)
        })
    }

    /// Get the most recently used key-value pair without removing it
    pub fn peek_mru(&self) -> Option<(&K, &V)> {
        self.head.map(|node_ptr| unsafe {
            let node = node_ptr.as_ptr();
            (&(*node).key, &(*node).value)
        })
    }

    // Internal: move a node to the front of the list
    fn move_to_front(&mut self, node_ptr: NonNull<Node<K, V>>) {
        if self.head == Some(node_ptr) {
            return; // Already at front
        }

        self.unlink(node_ptr);
        self.push_front(node_ptr);
    }

    // Internal: unlink a node from the list
    fn unlink(&mut self, node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_ptr();
            let prev = (*node).prev;
            let next = (*node).next;

            // Update prev's next pointer
            if let Some(prev_ptr) = prev {
                (*prev_ptr.as_ptr()).next = next;
            } else {
                // This was the head
                self.head = next;
            }

            // Update next's prev pointer
            if let Some(next_ptr) = next {
                (*next_ptr.as_ptr()).prev = prev;
            } else {
                // This was the tail
                self.tail = prev;
            }

            (*node).prev = None;
            (*node).next = None;
        }
    }

    // Internal: add a node at the front
    fn push_front(&mut self, node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_ptr();
            (*node).prev = None;
            (*node).next = self.head;

            if let Some(head_ptr) = self.head {
                (*head_ptr.as_ptr()).prev = Some(node_ptr);
            }

            self.head = Some(node_ptr);

            if self.tail.is_none() {
                self.tail = Some(node_ptr);
            }
        }
    }

    // Internal: evict the LRU item (tail)
    fn evict_lru(&mut self) {
        if let Some(tail_ptr) = self.tail {
            unsafe {
                let key = (*tail_ptr.as_ptr()).key.clone();
                self.map.remove(&key);
                self.unlink(tail_ptr);
                drop(Box::from_raw(tail_ptr.as_ptr()));
            }
        }
    }
}

impl<K, V> Drop for LruCache<K, V> {
    fn drop(&mut self) {
        let mut current = self.head;
        while let Some(node_ptr) = current {
            unsafe {
                current = (*node_ptr.as_ptr()).next;
                drop(Box::from_raw(node_ptr.as_ptr()));
            }
        }
    }
}

/// LRU Cache with TTL (Time To Live) support
/// Items expire after a specified duration
pub struct TtlLruCache<K, V> {
    inner: LruCache<K, (V, std::time::Instant)>,
    ttl: std::time::Duration,
}

impl<K, V> TtlLruCache<K, V>
where
    K: Hash + Eq + Clone,
{
    /// Create a new TTL LRU cache
    pub fn new(capacity: usize, ttl: std::time::Duration) -> Self {
        Self {
            inner: LruCache::new(capacity),
            ttl,
        }
    }

    /// Get a value if present and not expired
    pub fn get(&mut self, key: &K) -> Option<&V> {
        // Check if expired
        if let Some((_, timestamp)) = self.inner.peek(key)
            && timestamp.elapsed() > self.ttl
        {
            self.inner.remove(key);
            return None;
        }

        self.inner.get(key).map(|(v, _)| v)
    }

    /// Insert or update a key-value pair with current timestamp
    pub fn put(&mut self, key: K, value: V) {
        self.inner.put(key, (value, std::time::Instant::now()));
    }

    /// Remove a key from the cache
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.inner.remove(key).map(|(v, _)| v)
    }

    /// Remove all expired entries
    pub fn cleanup_expired(&mut self) {
        let now = std::time::Instant::now();
        let ttl = self.ttl;

        // Collect expired keys
        let expired: Vec<K> = self
            .inner
            .map
            .iter()
            .filter_map(|(k, &node_ptr)| {
                let (_, timestamp) = unsafe { &(*node_ptr.as_ptr()).value };
                if now.duration_since(*timestamp) > ttl {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect();

        // Remove expired entries
        for key in expired {
            self.inner.remove(&key);
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Simple LRU Cache using only standard library (no unsafe)
/// Slightly less efficient but safe implementation using Vec
pub struct SimpleLruCache<K, V> {
    capacity: usize,
    entries: Vec<(K, V)>, // Most recent at end
}

impl<K, V> SimpleLruCache<K, V>
where
    K: Eq,
{
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be positive");
        Self {
            capacity,
            entries: Vec::with_capacity(capacity),
        }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        let pos = self.entries.iter().position(|(k, _)| k == key)?;

        // Move to end (most recent)
        let entry = self.entries.remove(pos);
        self.entries.push(entry);

        self.entries.last().map(|(_, v)| v)
    }

    pub fn put(&mut self, key: K, value: V) -> Option<V> {
        // Check if key exists
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == &key) {
            let (_, old_value) = self.entries.remove(pos);
            self.entries.push((key, value));
            return Some(old_value);
        }

        // Evict if at capacity
        if self.entries.len() >= self.capacity {
            self.entries.remove(0); // Remove LRU (front)
        }

        self.entries.push((key, value));
        None
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let pos = self.entries.iter().position(|(k, _)| k == key)?;
        let (_, value) = self.entries.remove(pos);
        Some(value)
    }

    pub fn contains(&self, key: &K) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_basic_operations() {
        let mut cache: LruCache<i32, &str> = LruCache::new(3);

        cache.put(1, "one");
        cache.put(2, "two");
        cache.put(3, "three");

        assert_eq!(cache.get(&1), Some(&"one"));
        assert_eq!(cache.get(&2), Some(&"two"));
        assert_eq!(cache.get(&3), Some(&"three"));
        assert_eq!(cache.get(&4), None);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache: LruCache<i32, i32> = LruCache::new(2);

        cache.put(1, 10);
        cache.put(2, 20);
        cache.put(3, 30); // Should evict key 1

        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), Some(&20));
        assert_eq!(cache.get(&3), Some(&30));
    }

    #[test]
    fn test_lru_access_updates_order() {
        let mut cache: LruCache<i32, i32> = LruCache::new(2);

        cache.put(1, 10);
        cache.put(2, 20);
        cache.get(&1); // Access key 1, making it most recent
        cache.put(3, 30); // Should evict key 2, not key 1

        assert_eq!(cache.get(&1), Some(&10));
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.get(&3), Some(&30));
    }

    #[test]
    fn test_lru_update_value() {
        let mut cache: LruCache<i32, i32> = LruCache::new(2);

        cache.put(1, 10);
        let old = cache.put(1, 100);

        assert_eq!(old, Some(10));
        assert_eq!(cache.get(&1), Some(&100));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_remove() {
        let mut cache: LruCache<i32, i32> = LruCache::new(3);

        cache.put(1, 10);
        cache.put(2, 20);
        cache.put(3, 30);

        let removed = cache.remove(&2);
        assert_eq!(removed, Some(20));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&2), None);
    }

    #[test]
    fn test_lru_clear() {
        let mut cache: LruCache<i32, i32> = LruCache::new(3);

        cache.put(1, 10);
        cache.put(2, 20);
        cache.clear();

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_lru_peek() {
        let mut cache: LruCache<i32, i32> = LruCache::new(2);

        cache.put(1, 10);
        cache.put(2, 20);

        // Peek doesn't change order
        assert_eq!(cache.peek(&1), Some(&10));

        // Add new item - should evict 1 (still LRU since peek doesn't update)
        cache.put(3, 30);

        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), Some(&20));
    }

    #[test]
    fn test_lru_peek_lru_mru() {
        let mut cache: LruCache<i32, i32> = LruCache::new(3);

        cache.put(1, 10);
        cache.put(2, 20);
        cache.put(3, 30);

        assert_eq!(cache.peek_lru(), Some((&1, &10))); // Oldest
        assert_eq!(cache.peek_mru(), Some((&3, &30))); // Newest
    }

    #[test]
    fn test_lru_get_mut() {
        let mut cache: LruCache<i32, i32> = LruCache::new(2);

        cache.put(1, 10);

        if let Some(value) = cache.get_mut(&1) {
            *value = 100;
        }

        assert_eq!(cache.get(&1), Some(&100));
    }

    #[test]
    fn test_lru_strings() {
        let mut cache: LruCache<String, String> = LruCache::new(2);

        cache.put("key1".to_string(), "value1".to_string());
        cache.put("key2".to_string(), "value2".to_string());

        assert_eq!(cache.get(&"key1".to_string()), Some(&"value1".to_string()));
    }

    #[test]
    fn test_lru_capacity_one() {
        let mut cache: LruCache<i32, i32> = LruCache::new(1);

        cache.put(1, 10);
        assert_eq!(cache.get(&1), Some(&10));

        cache.put(2, 20);
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), Some(&20));
    }

    // Simple LRU Cache tests
    #[test]
    fn test_simple_lru_basic() {
        let mut cache: SimpleLruCache<i32, i32> = SimpleLruCache::new(2);

        cache.put(1, 10);
        cache.put(2, 20);

        assert_eq!(cache.get(&1), Some(&10));
        assert_eq!(cache.get(&2), Some(&20));
    }

    #[test]
    fn test_simple_lru_eviction() {
        let mut cache: SimpleLruCache<i32, i32> = SimpleLruCache::new(2);

        cache.put(1, 10);
        cache.put(2, 20);
        cache.get(&1); // Access 1
        cache.put(3, 30); // Evict 2

        assert_eq!(cache.get(&1), Some(&10));
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.get(&3), Some(&30));
    }

    // TTL Cache tests
    #[test]
    fn test_ttl_cache_basic() {
        let mut cache: TtlLruCache<i32, i32> =
            TtlLruCache::new(2, std::time::Duration::from_secs(60));

        cache.put(1, 10);
        assert_eq!(cache.get(&1), Some(&10));
    }

    #[test]
    fn test_ttl_cache_expiry() {
        let mut cache: TtlLruCache<i32, i32> =
            TtlLruCache::new(2, std::time::Duration::from_millis(50));

        cache.put(1, 10);
        assert_eq!(cache.get(&1), Some(&10));

        std::thread::sleep(std::time::Duration::from_millis(100));

        assert_eq!(cache.get(&1), None);
    }
}
