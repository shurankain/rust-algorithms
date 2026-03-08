// Skip List - Probabilistic alternative to balanced trees
//
// A skip list is a linked list with multiple levels of forward pointers.
// Higher levels skip over more elements, enabling O(log n) search on average.
//
// Time Complexity (expected):
// - Search: O(log n)
// - Insert: O(log n)
// - Delete: O(log n)
//
// Space Complexity: O(n) expected
//
// Advantages over balanced trees:
// - Simpler implementation
// - Lock-free concurrent versions possible
// - Cache-friendly sequential access
// - No rotations needed

use std::cmp::Ordering;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

const MAX_LEVEL: usize = 32;
const P: f64 = 0.5; // Probability for level promotion

type Link<K, V> = Option<NonNull<Node<K, V>>>;

struct Node<K, V> {
    key: MaybeUninit<K>,
    value: MaybeUninit<V>,
    forward: Vec<Link<K, V>>,
    is_sentinel: bool,
}

impl<K, V> Node<K, V> {
    fn new(key: K, value: V, level: usize) -> Self {
        Self {
            key: MaybeUninit::new(key),
            value: MaybeUninit::new(value),
            forward: vec![None; level + 1],
            is_sentinel: false,
        }
    }

    fn sentinel(level: usize) -> Self {
        Self {
            key: MaybeUninit::uninit(),
            value: MaybeUninit::uninit(),
            forward: vec![None; level],
            is_sentinel: true,
        }
    }

    // Helper to get forward pointer at level (avoids implicit autoref issues in Rust 2024)
    #[inline]
    fn get_forward(&self, level: usize) -> Link<K, V> {
        self.forward[level]
    }

    // Helper to set forward pointer at level
    #[inline]
    fn set_forward(&mut self, level: usize, link: Link<K, V>) {
        self.forward[level] = link;
    }

    #[inline]
    fn key(&self) -> &K {
        unsafe { self.key.assume_init_ref() }
    }

    #[inline]
    fn value(&self) -> &V {
        unsafe { self.value.assume_init_ref() }
    }

    #[inline]
    fn value_mut(&mut self) -> &mut V {
        unsafe { self.value.assume_init_mut() }
    }
}

/// Skip List - A probabilistic data structure for ordered key-value storage
pub struct SkipList<K, V> {
    head: NonNull<Node<K, V>>,
    level: usize,
    len: usize,
}

impl<K, V> SkipList<K, V>
where
    K: Ord,
{
    /// Create a new empty skip list
    pub fn new() -> Self {
        // Create sentinel head node
        let head = Box::new(Node::sentinel(MAX_LEVEL));

        Self {
            head: NonNull::new(Box::into_raw(head)).unwrap(),
            level: 0,
            len: 0,
        }
    }

    /// Insert a key-value pair. Returns the old value if key existed.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let mut update: [Link<K, V>; MAX_LEVEL] = [None; MAX_LEVEL];
        let mut current = self.head;

        // Find position and track update pointers
        unsafe {
            for i in (0..=self.level).rev() {
                while let Some(next) = (*current.as_ptr()).get_forward(i) {
                    if *(*next.as_ptr()).key() < key {
                        current = next;
                    } else {
                        break;
                    }
                }
                update[i] = Some(current);
            }

            // Check if key already exists
            if let Some(next) = (*current.as_ptr()).get_forward(0)
                && *(*next.as_ptr()).key() == key
            {
                // Update existing value
                let old_value = std::mem::replace((*next.as_ptr()).value_mut(), value);
                return Some(old_value);
            }
        }

        // Generate random level for new node
        let new_level = self.random_level();

        // Update level if needed
        if new_level > self.level {
            for update_item in update.iter_mut().take(new_level + 1).skip(self.level + 1) {
                *update_item = Some(self.head);
            }
            self.level = new_level;
        }

        // Create new node
        let new_node = Box::new(Node::new(key, value, new_level));
        let new_ptr = NonNull::new(Box::into_raw(new_node)).unwrap();

        // Insert node by updating forward pointers
        unsafe {
            for (i, update_item) in update.iter().enumerate().take(new_level + 1) {
                if let Some(update_node) = update_item {
                    (*new_ptr.as_ptr()).set_forward(i, (*update_node.as_ptr()).get_forward(i));
                    (*update_node.as_ptr()).set_forward(i, Some(new_ptr));
                }
            }
        }

        self.len += 1;
        None
    }

    /// Get a reference to the value associated with a key
    pub fn get(&self, key: &K) -> Option<&V> {
        let node = self.find_node(key)?;
        unsafe { Some((*node.as_ptr()).value()) }
    }

    /// Get a mutable reference to the value associated with a key
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let node = self.find_node(key)?;
        unsafe { Some((*node.as_ptr()).value_mut()) }
    }

    /// Check if the skip list contains a key
    pub fn contains(&self, key: &K) -> bool {
        self.find_node(key).is_some()
    }

    /// Remove a key from the skip list. Returns the value if present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let mut update: [Link<K, V>; MAX_LEVEL] = [None; MAX_LEVEL];
        let mut current = self.head;

        unsafe {
            // Find node and track update pointers
            for i in (0..=self.level).rev() {
                while let Some(next) = (*current.as_ptr()).get_forward(i) {
                    if *(*next.as_ptr()).key() < *key {
                        current = next;
                    } else {
                        break;
                    }
                }
                update[i] = Some(current);
            }

            // Check if key exists
            let target = (*current.as_ptr()).get_forward(0)?;
            if *(*target.as_ptr()).key() != *key {
                return None;
            }

            // Update forward pointers to skip the removed node
            for (i, update_item) in update.iter().enumerate().take(self.level + 1) {
                if let Some(update_node) = update_item
                    && (*update_node.as_ptr()).get_forward(i) == Some(target)
                {
                    (*update_node.as_ptr()).set_forward(i, (*target.as_ptr()).get_forward(i));
                }
            }

            // Decrease level if needed
            while self.level > 0 && (*self.head.as_ptr()).get_forward(self.level).is_none() {
                self.level -= 1;
            }

            // Free the node - extract value before dropping
            let removed_node = Box::from_raw(target.as_ptr());
            self.len -= 1;

            // Safety: the node was a real node (not sentinel) so value is initialized
            Some(removed_node.value.assume_init())
        }
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the skip list is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        unsafe {
            let mut current = (*self.head.as_ptr()).get_forward(0);
            while let Some(node) = current {
                let next = (*node.as_ptr()).get_forward(0);
                drop(Box::from_raw(node.as_ptr()));
                current = next;
            }

            // Reset head's forward pointers
            for i in 0..MAX_LEVEL {
                (*self.head.as_ptr()).set_forward(i, None);
            }
        }

        self.level = 0;
        self.len = 0;
    }

    /// Get the first (minimum) key-value pair
    pub fn first(&self) -> Option<(&K, &V)> {
        unsafe {
            let first = (*self.head.as_ptr()).get_forward(0)?;
            Some(((*first.as_ptr()).key(), (*first.as_ptr()).value()))
        }
    }

    /// Get the last (maximum) key-value pair
    pub fn last(&self) -> Option<(&K, &V)> {
        if self.is_empty() {
            return None;
        }

        unsafe {
            let mut current = self.head;

            // Traverse to the last node
            for i in (0..=self.level).rev() {
                while let Some(next) = (*current.as_ptr()).get_forward(i) {
                    current = next;
                }
            }

            if current == self.head {
                None
            } else {
                Some(((*current.as_ptr()).key(), (*current.as_ptr()).value()))
            }
        }
    }

    /// Iterate over all key-value pairs in order
    pub fn iter(&self) -> SkipListIter<'_, K, V> {
        unsafe {
            SkipListIter {
                current: (*self.head.as_ptr()).get_forward(0),
                _marker: std::marker::PhantomData,
            }
        }
    }

    /// Get range of entries with keys in [start, end)
    pub fn range(&self, start: &K, end: &K) -> Vec<(&K, &V)> {
        let mut result = Vec::new();

        // Find starting position
        let mut current = self.head;
        unsafe {
            for i in (0..=self.level).rev() {
                while let Some(next) = (*current.as_ptr()).get_forward(i) {
                    if *(*next.as_ptr()).key() < *start {
                        current = next;
                    } else {
                        break;
                    }
                }
            }

            // Collect elements in range
            current = match (*current.as_ptr()).get_forward(0) {
                Some(n) => n,
                None => return result,
            };

            while *(*current.as_ptr()).key() < *end {
                result.push(((*current.as_ptr()).key(), (*current.as_ptr()).value()));
                match (*current.as_ptr()).get_forward(0) {
                    Some(next) => current = next,
                    None => break,
                }
            }
        }

        result
    }

    /// Find the largest key less than or equal to the given key
    pub fn floor(&self, key: &K) -> Option<(&K, &V)> {
        let mut current = self.head;
        let mut result: Option<NonNull<Node<K, V>>> = None;

        unsafe {
            for i in (0..=self.level).rev() {
                while let Some(next) = (*current.as_ptr()).get_forward(i) {
                    match (*next.as_ptr()).key().cmp(key) {
                        Ordering::Less => {
                            current = next;
                            result = Some(next);
                        }
                        Ordering::Equal => {
                            return Some(((*next.as_ptr()).key(), (*next.as_ptr()).value()));
                        }
                        Ordering::Greater => break,
                    }
                }
            }

            result.map(|node| ((*node.as_ptr()).key(), (*node.as_ptr()).value()))
        }
    }

    /// Find the smallest key greater than or equal to the given key
    pub fn ceiling(&self, key: &K) -> Option<(&K, &V)> {
        let mut current = self.head;

        unsafe {
            for i in (0..=self.level).rev() {
                while let Some(next) = (*current.as_ptr()).get_forward(i) {
                    if *(*next.as_ptr()).key() < *key {
                        current = next;
                    } else {
                        break;
                    }
                }
            }

            let next = (*current.as_ptr()).get_forward(0)?;
            Some(((*next.as_ptr()).key(), (*next.as_ptr()).value()))
        }
    }

    // Internal: find the node with given key
    fn find_node(&self, key: &K) -> Option<NonNull<Node<K, V>>> {
        let mut current = self.head;

        unsafe {
            for i in (0..=self.level).rev() {
                while let Some(next) = (*current.as_ptr()).get_forward(i) {
                    if *(*next.as_ptr()).key() < *key {
                        current = next;
                    } else {
                        break;
                    }
                }
            }

            let next = (*current.as_ptr()).get_forward(0)?;
            if *(*next.as_ptr()).key() == *key {
                Some(next)
            } else {
                None
            }
        }
    }

    // Generate a random level for a new node
    fn random_level(&self) -> usize {
        let mut level = 0;
        // Simple PRNG for level generation
        let mut rng = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos() as u64;

        while level < MAX_LEVEL - 1 {
            // XorShift for quick random numbers
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;

            if (rng as f64 / u64::MAX as f64) < P {
                level += 1;
            } else {
                break;
            }
        }

        level
    }
}

impl<K, V> Default for SkipList<K, V>
where
    K: Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Drop for SkipList<K, V> {
    fn drop(&mut self) {
        unsafe {
            let mut current = (*self.head.as_ptr()).get_forward(0);
            while let Some(node_ptr) = current {
                let next = (*node_ptr.as_ptr()).get_forward(0);
                // Drop key and value manually since they're in MaybeUninit
                let node = Box::from_raw(node_ptr.as_ptr());
                if !node.is_sentinel {
                    std::ptr::drop_in_place(node.key.as_ptr() as *mut K);
                    std::ptr::drop_in_place(node.value.as_ptr() as *mut V);
                }
                current = next;
            }
            // Drop head node (sentinel - don't drop key/value)
            drop(Box::from_raw(self.head.as_ptr()));
        }
    }
}

/// Iterator over skip list entries
pub struct SkipListIter<'a, K, V> {
    current: Link<K, V>,
    _marker: std::marker::PhantomData<&'a (K, V)>,
}

impl<'a, K, V> Iterator for SkipListIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            let node = self.current?;
            self.current = (*node.as_ptr()).get_forward(0);
            Some(((*node.as_ptr()).key(), (*node.as_ptr()).value()))
        }
    }
}

/// Skip List Set - A set implementation using skip list
pub struct SkipListSet<K> {
    inner: SkipList<K, ()>,
}

impl<K> SkipListSet<K>
where
    K: Ord,
{
    pub fn new() -> Self {
        Self {
            inner: SkipList::new(),
        }
    }

    pub fn insert(&mut self, key: K) -> bool {
        self.inner.insert(key, ()).is_none()
    }

    pub fn contains(&self, key: &K) -> bool {
        self.inner.contains(key)
    }

    pub fn remove(&mut self, key: &K) -> bool {
        self.inner.remove(key).is_some()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn first(&self) -> Option<&K> {
        self.inner.first().map(|(k, _)| k)
    }

    pub fn last(&self) -> Option<&K> {
        self.inner.last().map(|(k, _)| k)
    }

    pub fn iter(&self) -> impl Iterator<Item = &K> {
        self.inner.iter().map(|(k, _)| k)
    }
}

impl<K: Ord> Default for SkipListSet<K> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skip_list_insert_get() {
        let mut list: SkipList<i32, &str> = SkipList::new();

        list.insert(3, "three");
        list.insert(1, "one");
        list.insert(2, "two");

        assert_eq!(list.get(&1), Some(&"one"));
        assert_eq!(list.get(&2), Some(&"two"));
        assert_eq!(list.get(&3), Some(&"three"));
        assert_eq!(list.get(&4), None);
    }

    #[test]
    fn test_skip_list_update() {
        let mut list: SkipList<i32, i32> = SkipList::new();

        assert!(list.insert(1, 10).is_none());
        assert_eq!(list.insert(1, 100), Some(10));
        assert_eq!(list.get(&1), Some(&100));
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_skip_list_remove() {
        let mut list: SkipList<i32, i32> = SkipList::new();

        list.insert(1, 10);
        list.insert(2, 20);
        list.insert(3, 30);

        assert_eq!(list.remove(&2), Some(20));
        assert_eq!(list.len(), 2);
        assert_eq!(list.get(&2), None);
        assert_eq!(list.get(&1), Some(&10));
        assert_eq!(list.get(&3), Some(&30));
    }

    #[test]
    fn test_skip_list_contains() {
        let mut list: SkipList<i32, i32> = SkipList::new();

        list.insert(1, 10);
        list.insert(2, 20);

        assert!(list.contains(&1));
        assert!(list.contains(&2));
        assert!(!list.contains(&3));
    }

    #[test]
    fn test_skip_list_len_empty() {
        let mut list: SkipList<i32, i32> = SkipList::new();

        assert!(list.is_empty());
        assert_eq!(list.len(), 0);

        list.insert(1, 10);
        assert!(!list.is_empty());
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_skip_list_clear() {
        let mut list: SkipList<i32, i32> = SkipList::new();

        list.insert(1, 10);
        list.insert(2, 20);
        list.clear();

        assert!(list.is_empty());
        assert_eq!(list.get(&1), None);
    }

    #[test]
    fn test_skip_list_first_last() {
        let mut list: SkipList<i32, i32> = SkipList::new();

        assert!(list.first().is_none());
        assert!(list.last().is_none());

        list.insert(5, 50);
        list.insert(1, 10);
        list.insert(9, 90);
        list.insert(3, 30);

        assert_eq!(list.first(), Some((&1, &10)));
        assert_eq!(list.last(), Some((&9, &90)));
    }

    #[test]
    fn test_skip_list_iter() {
        let mut list: SkipList<i32, i32> = SkipList::new();

        list.insert(3, 30);
        list.insert(1, 10);
        list.insert(2, 20);

        let items: Vec<_> = list.iter().collect();
        assert_eq!(items, vec![(&1, &10), (&2, &20), (&3, &30)]);
    }

    #[test]
    fn test_skip_list_range() {
        let mut list: SkipList<i32, i32> = SkipList::new();

        for i in 1..=10 {
            list.insert(i, i * 10);
        }

        let range: Vec<_> = list.range(&3, &7);
        assert_eq!(range, vec![(&3, &30), (&4, &40), (&5, &50), (&6, &60)]);
    }

    #[test]
    fn test_skip_list_floor_ceiling() {
        let mut list: SkipList<i32, i32> = SkipList::new();

        list.insert(2, 20);
        list.insert(4, 40);
        list.insert(6, 60);
        list.insert(8, 80);

        assert_eq!(list.floor(&5), Some((&4, &40)));
        assert_eq!(list.floor(&4), Some((&4, &40)));
        assert_eq!(list.floor(&1), None);

        assert_eq!(list.ceiling(&5), Some((&6, &60)));
        assert_eq!(list.ceiling(&6), Some((&6, &60)));
        assert_eq!(list.ceiling(&9), None);
    }

    #[test]
    fn test_skip_list_get_mut() {
        let mut list: SkipList<i32, i32> = SkipList::new();

        list.insert(1, 10);

        if let Some(v) = list.get_mut(&1) {
            *v = 100;
        }

        assert_eq!(list.get(&1), Some(&100));
    }

    #[test]
    fn test_skip_list_strings() {
        let mut list: SkipList<String, String> = SkipList::new();

        list.insert("b".to_string(), "beta".to_string());
        list.insert("a".to_string(), "alpha".to_string());
        list.insert("c".to_string(), "gamma".to_string());

        assert_eq!(list.get(&"a".to_string()), Some(&"alpha".to_string()));
        assert_eq!(list.first().map(|(k, _)| k.as_str()), Some("a"));
    }

    #[test]
    fn test_skip_list_many_elements() {
        let mut list: SkipList<i32, i32> = SkipList::new();

        for i in 0..1000 {
            list.insert(i, i * 2);
        }

        assert_eq!(list.len(), 1000);

        for i in 0..1000 {
            assert_eq!(list.get(&i), Some(&(i * 2)));
        }

        // Remove half
        for i in (0..1000).step_by(2) {
            list.remove(&i);
        }

        assert_eq!(list.len(), 500);
    }

    // Skip List Set tests
    #[test]
    fn test_skip_list_set_basic() {
        let mut set: SkipListSet<i32> = SkipListSet::new();

        assert!(set.insert(1));
        assert!(set.insert(2));
        assert!(!set.insert(1)); // Duplicate

        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(!set.contains(&3));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_skip_list_set_remove() {
        let mut set: SkipListSet<i32> = SkipListSet::new();

        set.insert(1);
        set.insert(2);

        assert!(set.remove(&1));
        assert!(!set.remove(&1)); // Already removed
        assert!(!set.contains(&1));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_skip_list_set_iter() {
        let mut set: SkipListSet<i32> = SkipListSet::new();

        set.insert(3);
        set.insert(1);
        set.insert(2);

        let items: Vec<_> = set.iter().copied().collect();
        assert_eq!(items, vec![1, 2, 3]);
    }
}
