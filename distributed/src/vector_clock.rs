// Vector Clocks
//
// A mechanism for tracking causality and ordering events in distributed systems.
// Developed by Leslie Lamport (1978) and extended by Fidge and Mattern (1988).
//
// Key concepts:
// - Each process maintains a vector of logical timestamps (one per process)
// - On local event: increment own counter
// - On send: increment own counter, attach clock to message
// - On receive: merge with received clock (take max), then increment own counter
//
// Properties:
// - Captures "happens-before" relationship (causal ordering)
// - Can detect concurrent events (neither happened before the other)
// - Total ordering possible via tie-breaking (but arbitrary for concurrent events)
//
// Comparison:
// - VC(a) < VC(b): event a happened before event b
// - VC(a) || VC(b): events a and b are concurrent (no causal relationship)
// - VC(a) == VC(b): same event (typically)
//
// Used in:
// - Distributed databases (conflict detection)
// - Version control systems
// - Eventual consistency systems (Amazon Dynamo)
// - CRDTs (Conflict-free Replicated Data Types)

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;

/// Type alias for process/node identifier
pub type ProcessId = u64;

/// Type alias for logical timestamp
pub type Timestamp = u64;

/// A vector clock for tracking causality in distributed systems
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorClock {
    /// The clock vector: process_id -> timestamp
    clock: HashMap<ProcessId, Timestamp>,
}

impl Default for VectorClock {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorClock {
    /// Create a new empty vector clock
    pub fn new() -> Self {
        Self {
            clock: HashMap::new(),
        }
    }

    /// Create a vector clock with initial processes
    pub fn with_processes(processes: &[ProcessId]) -> Self {
        let mut clock = HashMap::new();
        for &p in processes {
            clock.insert(p, 0);
        }
        Self { clock }
    }

    /// Get the timestamp for a specific process
    pub fn get(&self, process: ProcessId) -> Timestamp {
        *self.clock.get(&process).unwrap_or(&0)
    }

    /// Set the timestamp for a specific process
    pub fn set(&mut self, process: ProcessId, timestamp: Timestamp) {
        self.clock.insert(process, timestamp);
    }

    /// Increment the clock for a process (local event)
    pub fn increment(&mut self, process: ProcessId) {
        let current = self.get(process);
        self.clock.insert(process, current + 1);
    }

    /// Record a local event for a process
    /// Alias for increment
    pub fn tick(&mut self, process: ProcessId) {
        self.increment(process);
    }

    /// Prepare clock for sending a message
    /// Increments the sender's clock and returns a copy to attach to message
    pub fn send(&mut self, sender: ProcessId) -> VectorClock {
        self.increment(sender);
        self.clone()
    }

    /// Receive a message with attached vector clock
    /// Merges the received clock and increments the receiver's clock
    pub fn receive(&mut self, receiver: ProcessId, received: &VectorClock) {
        self.merge(received);
        self.increment(receiver);
    }

    /// Merge another vector clock into this one (take max of each component)
    pub fn merge(&mut self, other: &VectorClock) {
        for (&process, &timestamp) in &other.clock {
            let current = self.get(process);
            if timestamp > current {
                self.clock.insert(process, timestamp);
            }
        }
    }

    /// Check if this clock is less than another (happened before)
    /// Returns true if this clock is strictly less than other
    /// (all components <= and at least one <)
    pub fn happened_before(&self, other: &VectorClock) -> bool {
        let mut at_least_one_less = false;

        // Check all processes in both clocks
        let all_processes: std::collections::HashSet<ProcessId> = self
            .clock
            .keys()
            .chain(other.clock.keys())
            .copied()
            .collect();

        for process in all_processes {
            let self_ts = self.get(process);
            let other_ts = other.get(process);

            if self_ts > other_ts {
                return false; // Not less than or equal
            }
            if self_ts < other_ts {
                at_least_one_less = true;
            }
        }

        at_least_one_less
    }

    /// Check if this clock happened after another
    pub fn happened_after(&self, other: &VectorClock) -> bool {
        other.happened_before(self)
    }

    /// Check if two clocks are concurrent (neither happened before the other)
    pub fn concurrent_with(&self, other: &VectorClock) -> bool {
        !self.happened_before(other) && !other.happened_before(self) && self != other
    }

    /// Compare two vector clocks
    /// Returns Some(Ordering) if comparable, None if concurrent
    pub fn partial_compare(&self, other: &VectorClock) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else if self.happened_before(other) {
            Some(Ordering::Less)
        } else if other.happened_before(self) {
            Some(Ordering::Greater)
        } else {
            None // Concurrent
        }
    }

    /// Get all processes in this clock
    pub fn processes(&self) -> impl Iterator<Item = ProcessId> + '_ {
        self.clock.keys().copied()
    }

    /// Get the number of processes tracked
    pub fn num_processes(&self) -> usize {
        self.clock.len()
    }

    /// Check if the clock is empty (no events recorded)
    pub fn is_empty(&self) -> bool {
        self.clock.values().all(|&t| t == 0)
    }

    /// Get the maximum timestamp in the clock
    pub fn max_timestamp(&self) -> Timestamp {
        self.clock.values().copied().max().unwrap_or(0)
    }

    /// Get the sum of all timestamps (total events)
    pub fn total_events(&self) -> Timestamp {
        self.clock.values().sum()
    }

    /// Create a copy of the internal clock map
    pub fn to_map(&self) -> HashMap<ProcessId, Timestamp> {
        self.clock.clone()
    }
}

impl fmt::Display for VectorClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut entries: Vec<_> = self.clock.iter().collect();
        entries.sort_by_key(|&(&k, _)| k);

        write!(f, "[")?;
        for (i, &(&process, &timestamp)) in entries.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}:{}", process, timestamp)?;
        }
        write!(f, "]")
    }
}

/// An event with an associated vector clock timestamp
#[derive(Debug, Clone)]
pub struct Event<T> {
    /// The process that generated this event
    pub process: ProcessId,
    /// The event data
    pub data: T,
    /// The vector clock at the time of the event
    pub clock: VectorClock,
}

impl<T> Event<T> {
    /// Create a new event
    pub fn new(process: ProcessId, data: T, clock: VectorClock) -> Self {
        Self {
            process,
            data,
            clock,
        }
    }

    /// Check if this event happened before another
    pub fn happened_before<U>(&self, other: &Event<U>) -> bool {
        self.clock.happened_before(&other.clock)
    }

    /// Check if this event is concurrent with another
    pub fn concurrent_with<U>(&self, other: &Event<U>) -> bool {
        self.clock.concurrent_with(&other.clock)
    }
}

/// A process in a distributed system with a vector clock
#[derive(Debug, Clone)]
pub struct Process<T: Clone> {
    /// Process identifier
    id: ProcessId,
    /// Current vector clock
    clock: VectorClock,
    /// Event log
    events: Vec<Event<T>>,
}

impl<T: Clone> Process<T> {
    /// Create a new process
    pub fn new(id: ProcessId) -> Self {
        Self {
            id,
            clock: VectorClock::new(),
            events: Vec::new(),
        }
    }

    /// Create a new process with known peers
    pub fn with_peers(id: ProcessId, peers: &[ProcessId]) -> Self {
        let mut all = vec![id];
        all.extend(peers);
        Self {
            id,
            clock: VectorClock::with_processes(&all),
            events: Vec::new(),
        }
    }

    /// Get the process ID
    pub fn id(&self) -> ProcessId {
        self.id
    }

    /// Get the current vector clock
    pub fn clock(&self) -> &VectorClock {
        &self.clock
    }

    /// Get all events
    pub fn events(&self) -> &[Event<T>] {
        &self.events
    }

    /// Record a local event
    pub fn local_event(&mut self, data: T) -> &Event<T> {
        self.clock.tick(self.id);
        let event = Event::new(self.id, data, self.clock.clone());
        self.events.push(event);
        self.events.last().unwrap()
    }

    /// Prepare to send a message
    /// Returns the clock to attach to the message
    pub fn prepare_send(&mut self) -> VectorClock {
        self.clock.send(self.id)
    }

    /// Send a message (combines prepare_send with creating an event)
    pub fn send_message(&mut self, data: T) -> (VectorClock, &Event<T>) {
        let msg_clock = self.clock.send(self.id);
        let event = Event::new(self.id, data, self.clock.clone());
        self.events.push(event);
        (msg_clock, self.events.last().unwrap())
    }

    /// Receive a message with attached clock
    pub fn receive_message(&mut self, data: T, received_clock: &VectorClock) -> &Event<T> {
        self.clock.receive(self.id, received_clock);
        let event = Event::new(self.id, data, self.clock.clone());
        self.events.push(event);
        self.events.last().unwrap()
    }
}

/// Compare two events for causal ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalOrder {
    /// First event happened before second
    Before,
    /// First event happened after second
    After,
    /// Events are concurrent
    Concurrent,
    /// Events have the same clock (same event)
    Equal,
}

/// Compare two events causally
pub fn compare_events<T, U>(a: &Event<T>, b: &Event<U>) -> CausalOrder {
    match a.clock.partial_compare(&b.clock) {
        Some(Ordering::Less) => CausalOrder::Before,
        Some(Ordering::Greater) => CausalOrder::After,
        Some(Ordering::Equal) => CausalOrder::Equal,
        None => CausalOrder::Concurrent,
    }
}

/// Sort events by causal order (topological sort)
/// Returns events in an order where causes come before effects
/// Concurrent events maintain their relative order
pub fn causal_sort<T: Clone>(events: &[Event<T>]) -> Vec<Event<T>> {
    let mut sorted: Vec<Event<T>> = events.to_vec();

    // Simple insertion sort based on causality
    for i in 1..sorted.len() {
        let mut j = i;
        while j > 0 && sorted[j].happened_before(&sorted[j - 1]) {
            sorted.swap(j, j - 1);
            j -= 1;
        }
    }

    sorted
}

/// Find all events that are concurrent with a given event
pub fn find_concurrent<T: Clone>(event: &Event<T>, events: &[Event<T>]) -> Vec<usize> {
    events
        .iter()
        .enumerate()
        .filter(|(_, e)| event.concurrent_with(e))
        .map(|(i, _)| i)
        .collect()
}

/// Build a partial order graph from events
/// Returns edges where (a, b) means a happened before b
pub fn build_causality_graph<T>(events: &[Event<T>]) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();

    for i in 0..events.len() {
        for j in 0..events.len() {
            if i != j && events[i].happened_before(&events[j]) {
                // Only add direct edges (no transitivity reduction for simplicity)
                edges.push((i, j));
            }
        }
    }

    edges
}

/// Statistics about a set of vector clocks
#[derive(Debug, Clone, Default)]
pub struct VectorClockStats {
    /// Number of clocks analyzed
    pub num_clocks: usize,
    /// Total causal relationships (happened-before pairs)
    pub causal_pairs: usize,
    /// Total concurrent pairs
    pub concurrent_pairs: usize,
    /// Maximum clock size (number of processes)
    pub max_processes: usize,
    /// Maximum timestamp seen
    pub max_timestamp: Timestamp,
}

/// Analyze a set of events for statistics
pub fn analyze_events<T>(events: &[Event<T>]) -> VectorClockStats {
    let mut stats = VectorClockStats {
        num_clocks: events.len(),
        ..Default::default()
    };

    for event in events {
        stats.max_processes = stats.max_processes.max(event.clock.num_processes());
        stats.max_timestamp = stats.max_timestamp.max(event.clock.max_timestamp());
    }

    // Count causal and concurrent pairs
    for i in 0..events.len() {
        for j in (i + 1)..events.len() {
            if events[i].happened_before(&events[j]) || events[j].happened_before(&events[i]) {
                stats.causal_pairs += 1;
            } else if events[i].concurrent_with(&events[j]) {
                stats.concurrent_pairs += 1;
            }
        }
    }

    stats
}

/// Dotted Version Vector (DVV) - optimized for client-server scenarios
/// Tracks both the server's version and the client's "dot"
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DottedVersionVector {
    /// The base version vector
    pub base: VectorClock,
    /// The dot: (process, timestamp) of the last update
    pub dot: Option<(ProcessId, Timestamp)>,
}

impl DottedVersionVector {
    /// Create a new empty DVV
    pub fn new() -> Self {
        Self {
            base: VectorClock::new(),
            dot: None,
        }
    }

    /// Create from a vector clock with a dot
    pub fn from_clock(clock: VectorClock, process: ProcessId) -> Self {
        let timestamp = clock.get(process);
        Self {
            base: clock,
            dot: Some((process, timestamp)),
        }
    }

    /// Advance the DVV for a write by a process
    pub fn advance(&mut self, process: ProcessId) {
        self.base.increment(process);
        let new_ts = self.base.get(process);
        self.dot = Some((process, new_ts));
    }

    /// Merge with another DVV
    pub fn merge(&mut self, other: &DottedVersionVector) {
        self.base.merge(&other.base);
        // After merge, dot becomes invalid (represents merged state)
        self.dot = None;
    }

    /// Check if this DVV descends from another (is newer)
    pub fn descends_from(&self, other: &DottedVersionVector) -> bool {
        self.base.happened_after(&other.base) || self.base == other.base
    }

    /// Check if this DVV is concurrent with another
    pub fn concurrent_with(&self, other: &DottedVersionVector) -> bool {
        self.base.concurrent_with(&other.base)
    }
}

impl Default for DottedVersionVector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_clock_creation() {
        let vc = VectorClock::new();
        assert!(vc.is_empty());
        assert_eq!(vc.num_processes(), 0);

        let vc2 = VectorClock::with_processes(&[1, 2, 3]);
        assert_eq!(vc2.num_processes(), 3);
        assert_eq!(vc2.get(1), 0);
        assert_eq!(vc2.get(2), 0);
        assert_eq!(vc2.get(3), 0);
    }

    #[test]
    fn test_increment() {
        let mut vc = VectorClock::new();

        vc.increment(1);
        assert_eq!(vc.get(1), 1);
        assert_eq!(vc.get(2), 0);

        vc.increment(1);
        assert_eq!(vc.get(1), 2);

        vc.increment(2);
        assert_eq!(vc.get(2), 1);
    }

    #[test]
    fn test_merge() {
        let mut vc1 = VectorClock::new();
        vc1.set(1, 3);
        vc1.set(2, 1);

        let mut vc2 = VectorClock::new();
        vc2.set(1, 2);
        vc2.set(2, 4);
        vc2.set(3, 1);

        vc1.merge(&vc2);

        assert_eq!(vc1.get(1), 3); // max(3, 2)
        assert_eq!(vc1.get(2), 4); // max(1, 4)
        assert_eq!(vc1.get(3), 1); // new process from vc2
    }

    #[test]
    fn test_happened_before() {
        let mut vc1 = VectorClock::new();
        vc1.set(1, 1);
        vc1.set(2, 0);

        let mut vc2 = VectorClock::new();
        vc2.set(1, 2);
        vc2.set(2, 1);

        assert!(vc1.happened_before(&vc2));
        assert!(!vc2.happened_before(&vc1));
    }

    #[test]
    fn test_concurrent() {
        let mut vc1 = VectorClock::new();
        vc1.set(1, 2);
        vc1.set(2, 1);

        let mut vc2 = VectorClock::new();
        vc2.set(1, 1);
        vc2.set(2, 2);

        assert!(!vc1.happened_before(&vc2));
        assert!(!vc2.happened_before(&vc1));
        assert!(vc1.concurrent_with(&vc2));
    }

    #[test]
    fn test_send_receive() {
        let mut sender = VectorClock::with_processes(&[1, 2]);
        let mut receiver = VectorClock::with_processes(&[1, 2]);

        // Sender does some local work
        sender.tick(1);
        sender.tick(1);

        // Sender sends message
        let msg_clock = sender.send(1);
        assert_eq!(sender.get(1), 3);

        // Receiver does some local work
        receiver.tick(2);

        // Receiver receives message
        receiver.receive(2, &msg_clock);

        // Receiver should have merged clocks and incremented
        assert_eq!(receiver.get(1), 3); // From sender
        assert_eq!(receiver.get(2), 2); // Own tick + receive
    }

    #[test]
    fn test_partial_compare() {
        let mut vc1 = VectorClock::new();
        vc1.set(1, 1);

        let mut vc2 = VectorClock::new();
        vc2.set(1, 2);

        assert_eq!(vc1.partial_compare(&vc2), Some(Ordering::Less));
        assert_eq!(vc2.partial_compare(&vc1), Some(Ordering::Greater));

        let vc3 = vc1.clone();
        assert_eq!(vc1.partial_compare(&vc3), Some(Ordering::Equal));

        let mut vc4 = VectorClock::new();
        vc4.set(2, 1);
        assert_eq!(vc1.partial_compare(&vc4), None); // Concurrent
    }

    #[test]
    fn test_process() {
        let mut p1 = Process::<String>::new(1);
        let mut p2 = Process::<String>::new(2);

        // P1 does local event
        p1.local_event("event1".to_string());
        assert_eq!(p1.clock().get(1), 1);

        // P1 sends to P2
        let (msg_clock, _) = p1.send_message("hello".to_string());

        // P2 receives
        p2.receive_message("hello received".to_string(), &msg_clock);

        // P2's clock should reflect the message
        assert_eq!(p2.clock().get(1), 2); // From P1's send
        assert_eq!(p2.clock().get(2), 1); // P2's receive
    }

    #[test]
    fn test_event_ordering() {
        let mut p1 = Process::<&str>::new(1);
        let mut p2 = Process::<&str>::new(2);

        // Create events
        p1.local_event("e1");
        let (msg, _) = p1.send_message("m1");
        p2.receive_message("m1_recv", &msg);
        p2.local_event("e2");

        let events = p1
            .events()
            .iter()
            .chain(p2.events().iter())
            .cloned()
            .collect::<Vec<_>>();

        // e1 happened before m1_recv
        assert!(events[0].happened_before(&events[2]));

        // m1 happened before m1_recv (send before receive)
        assert!(events[1].happened_before(&events[2]));
    }

    #[test]
    fn test_causal_order_enum() {
        let mut vc1 = VectorClock::new();
        vc1.set(1, 1);

        let mut vc2 = VectorClock::new();
        vc2.set(1, 2);

        let e1 = Event::new(1, "a", vc1.clone());
        let e2 = Event::new(1, "b", vc2);

        assert_eq!(compare_events(&e1, &e2), CausalOrder::Before);
        assert_eq!(compare_events(&e2, &e1), CausalOrder::After);

        let e3 = Event::new(1, "c", vc1.clone());
        assert_eq!(compare_events(&e1, &e3), CausalOrder::Equal);

        // Concurrent events
        let mut vc3 = VectorClock::new();
        vc3.set(2, 1);
        let e4 = Event::new(2, "d", vc3);
        assert_eq!(compare_events(&e1, &e4), CausalOrder::Concurrent);
    }

    #[test]
    fn test_causal_sort() {
        let mut vc1 = VectorClock::new();
        vc1.set(1, 1);

        let mut vc2 = VectorClock::new();
        vc2.set(1, 2);

        let mut vc3 = VectorClock::new();
        vc3.set(1, 3);

        // Create events out of order
        let e2 = Event::new(1, "b", vc2);
        let e3 = Event::new(1, "c", vc3);
        let e1 = Event::new(1, "a", vc1);

        let events = vec![e2, e3, e1];
        let sorted = causal_sort(&events);

        assert_eq!(sorted[0].data, "a");
        assert_eq!(sorted[1].data, "b");
        assert_eq!(sorted[2].data, "c");
    }

    #[test]
    fn test_find_concurrent() {
        let mut vc1 = VectorClock::new();
        vc1.set(1, 1);

        let mut vc2 = VectorClock::new();
        vc2.set(2, 1);

        let mut vc3 = VectorClock::new();
        vc3.set(1, 2);
        vc3.set(2, 1);

        let e1 = Event::new(1, "a", vc1.clone());
        let e2 = Event::new(2, "b", vc2);
        let e3 = Event::new(1, "c", vc3);

        let events = vec![e1.clone(), e2, e3];
        let concurrent = find_concurrent(&e1, &events);

        // e1 is concurrent with e2 (index 1)
        assert!(concurrent.contains(&1));
        // e1 is not concurrent with itself or e3
        assert!(!concurrent.contains(&0));
        assert!(!concurrent.contains(&2));
    }

    #[test]
    fn test_display() {
        let mut vc = VectorClock::new();
        vc.set(1, 3);
        vc.set(2, 5);

        let s = format!("{}", vc);
        assert!(s.contains("1:3"));
        assert!(s.contains("2:5"));
    }

    #[test]
    fn test_statistics() {
        let mut p1 = Process::<i32>::new(1);
        let mut p2 = Process::<i32>::new(2);

        p1.local_event(1);
        let (msg, _) = p1.send_message(2);
        p2.receive_message(3, &msg);
        p2.local_event(4);
        p1.local_event(5);

        let all_events: Vec<Event<i32>> = p1
            .events()
            .iter()
            .chain(p2.events().iter())
            .cloned()
            .collect();

        let stats = analyze_events(&all_events);
        assert_eq!(stats.num_clocks, 5);
        assert!(stats.causal_pairs > 0);
    }

    #[test]
    fn test_dotted_version_vector() {
        let mut dvv = DottedVersionVector::new();
        assert!(dvv.dot.is_none());

        dvv.advance(1);
        assert_eq!(dvv.dot, Some((1, 1)));
        assert_eq!(dvv.base.get(1), 1);

        dvv.advance(1);
        assert_eq!(dvv.dot, Some((1, 2)));
    }

    #[test]
    fn test_dvv_merge() {
        let mut dvv1 = DottedVersionVector::new();
        dvv1.advance(1);
        dvv1.advance(1);

        let mut dvv2 = DottedVersionVector::new();
        dvv2.advance(2);

        dvv1.merge(&dvv2);
        assert_eq!(dvv1.base.get(1), 2);
        assert_eq!(dvv1.base.get(2), 1);
        assert!(dvv1.dot.is_none()); // Dot invalidated after merge
    }

    #[test]
    fn test_dvv_descends_from() {
        let mut dvv1 = DottedVersionVector::new();
        dvv1.advance(1);

        let mut dvv2 = dvv1.clone();
        dvv2.advance(1);

        assert!(dvv2.descends_from(&dvv1));
        assert!(!dvv1.descends_from(&dvv2));
    }

    #[test]
    fn test_max_timestamp() {
        let mut vc = VectorClock::new();
        vc.set(1, 5);
        vc.set(2, 3);
        vc.set(3, 7);

        assert_eq!(vc.max_timestamp(), 7);
    }

    #[test]
    fn test_total_events() {
        let mut vc = VectorClock::new();
        vc.set(1, 5);
        vc.set(2, 3);
        vc.set(3, 7);

        assert_eq!(vc.total_events(), 15);
    }

    #[test]
    fn test_build_causality_graph() {
        let mut vc1 = VectorClock::new();
        vc1.set(1, 1);

        let mut vc2 = VectorClock::new();
        vc2.set(1, 2);

        let mut vc3 = VectorClock::new();
        vc3.set(2, 1);

        let e1 = Event::new(1, "a", vc1);
        let e2 = Event::new(1, "b", vc2);
        let e3 = Event::new(2, "c", vc3);

        let events = vec![e1, e2, e3];
        let graph = build_causality_graph(&events);

        // e1 -> e2 should be in the graph
        assert!(graph.contains(&(0, 1)));
        // e1 and e3 are concurrent, no edge
        assert!(!graph.contains(&(0, 2)));
        assert!(!graph.contains(&(2, 0)));
    }

    #[test]
    fn test_three_process_scenario() {
        let mut p1 = Process::<&str>::with_peers(1, &[2, 3]);
        let mut p2 = Process::<&str>::with_peers(2, &[1, 3]);
        let mut p3 = Process::<&str>::with_peers(3, &[1, 2]);

        // P1 sends to P2
        let (m1, _) = p1.send_message("p1->p2");
        p2.receive_message("recv_m1", &m1);

        // P2 sends to P3
        let (m2, _) = p2.send_message("p2->p3");
        p3.receive_message("recv_m2", &m2);

        // P3's clock should know about all previous events
        assert!(p3.clock().get(1) > 0);
        assert!(p3.clock().get(2) > 0);
        assert!(p3.clock().get(3) > 0);
    }

    #[test]
    fn test_empty_clock_comparison() {
        let vc1 = VectorClock::new();
        let vc2 = VectorClock::new();

        assert_eq!(vc1, vc2);
        assert!(!vc1.happened_before(&vc2));
        assert!(!vc1.concurrent_with(&vc2));
    }
}
