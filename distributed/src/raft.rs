// Raft Consensus Algorithm (Simplified)
//
// Raft is a consensus algorithm designed to be understandable.
// It's equivalent to Paxos in fault-tolerance and performance but
// decomposed into relatively independent subproblems.
//
// Key concepts:
// 1. Leader Election: One node is elected leader, handles all client requests
// 2. Log Replication: Leader replicates log entries to followers
// 3. Safety: If any server has applied a log entry, no other server will
//    apply a different command for the same index
//
// Node states:
// - Follower: Passive, responds to RPCs from leader/candidates
// - Candidate: Active, requests votes to become leader
// - Leader: Handles all client requests, replicates log
//
// This is a simplified educational implementation:
// - Single-threaded simulation (no actual network)
// - Synchronous message passing
// - Focus on core algorithm, not production concerns
//
// References:
// - "In Search of an Understandable Consensus Algorithm" (Ongaro & Ousterhout, 2014)
// - https://raft.github.io/

use std::collections::{HashMap, HashSet, VecDeque};

/// Unique identifier for a node in the cluster
pub type NodeId = u64;

/// Term number (monotonically increasing)
pub type Term = u64;

/// Log entry index (1-based)
pub type LogIndex = u64;

/// The state a Raft node can be in
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    Follower,
    Candidate,
    Leader,
}

/// A log entry in the Raft log
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogEntry<T: Clone> {
    /// The term when entry was received by leader
    pub term: Term,
    /// The index of this entry
    pub index: LogIndex,
    /// The command to be applied to state machine
    pub command: T,
}

/// Request for votes (sent by candidates)
#[derive(Debug, Clone)]
pub struct RequestVote {
    /// Candidate's term
    pub term: Term,
    /// Candidate requesting vote
    pub candidate_id: NodeId,
    /// Index of candidate's last log entry
    pub last_log_index: LogIndex,
    /// Term of candidate's last log entry
    pub last_log_term: Term,
}

/// Response to RequestVote
#[derive(Debug, Clone)]
pub struct RequestVoteResponse {
    /// Current term, for candidate to update itself
    pub term: Term,
    /// True if candidate received vote
    pub vote_granted: bool,
}

/// Append entries RPC (sent by leader)
#[derive(Debug, Clone)]
pub struct AppendEntries<T: Clone> {
    /// Leader's term
    pub term: Term,
    /// Leader's ID so follower can redirect clients
    pub leader_id: NodeId,
    /// Index of log entry immediately preceding new ones
    pub prev_log_index: LogIndex,
    /// Term of prev_log_index entry
    pub prev_log_term: Term,
    /// Log entries to store (empty for heartbeat)
    pub entries: Vec<LogEntry<T>>,
    /// Leader's commit index
    pub leader_commit: LogIndex,
}

/// Response to AppendEntries
#[derive(Debug, Clone)]
pub struct AppendEntriesResponse {
    /// Current term, for leader to update itself
    pub term: Term,
    /// True if follower contained entry matching prev_log_index and prev_log_term
    pub success: bool,
    /// The index of the last entry the follower has (for optimization)
    pub match_index: LogIndex,
}

/// Messages that can be sent between nodes
#[derive(Debug, Clone)]
pub enum Message<T: Clone> {
    RequestVote(RequestVote),
    RequestVoteResponse(RequestVoteResponse),
    AppendEntries(AppendEntries<T>),
    AppendEntriesResponse(AppendEntriesResponse),
}

/// Persistent state on all servers (survives crashes)
#[derive(Debug, Clone)]
pub struct PersistentState<T: Clone> {
    /// Latest term server has seen
    pub current_term: Term,
    /// Candidate that received vote in current term (or None)
    pub voted_for: Option<NodeId>,
    /// Log entries
    pub log: Vec<LogEntry<T>>,
}

impl<T: Clone> Default for PersistentState<T> {
    fn default() -> Self {
        Self {
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
        }
    }
}

/// Volatile state on all servers
#[derive(Debug, Clone, Default)]
pub struct VolatileState {
    /// Index of highest log entry known to be committed
    pub commit_index: LogIndex,
    /// Index of highest log entry applied to state machine
    pub last_applied: LogIndex,
}

/// Volatile state on leaders (reinitialized after election)
#[derive(Debug, Clone)]
pub struct LeaderState {
    /// For each server, index of next log entry to send
    pub next_index: HashMap<NodeId, LogIndex>,
    /// For each server, index of highest log entry known to be replicated
    pub match_index: HashMap<NodeId, LogIndex>,
}

impl LeaderState {
    fn new(last_log_index: LogIndex, peers: &[NodeId]) -> Self {
        let mut next_index = HashMap::new();
        let mut match_index = HashMap::new();
        for &peer in peers {
            next_index.insert(peer, last_log_index + 1);
            match_index.insert(peer, 0);
        }
        Self {
            next_index,
            match_index,
        }
    }
}

/// Configuration for the Raft node
#[derive(Debug, Clone)]
pub struct RaftConfig {
    /// This node's ID
    pub id: NodeId,
    /// IDs of all peers (not including self)
    pub peers: Vec<NodeId>,
    /// Election timeout range (min ticks)
    pub election_timeout_min: u64,
    /// Election timeout range (max ticks)
    pub election_timeout_max: u64,
    /// Heartbeat interval (ticks)
    pub heartbeat_interval: u64,
}

impl RaftConfig {
    /// Create a new configuration
    pub fn new(id: NodeId, peers: Vec<NodeId>) -> Self {
        Self {
            id,
            peers,
            election_timeout_min: 150,
            election_timeout_max: 300,
            heartbeat_interval: 50,
        }
    }

    /// Total number of nodes in the cluster
    pub fn cluster_size(&self) -> usize {
        self.peers.len() + 1
    }

    /// Majority needed for consensus
    pub fn majority(&self) -> usize {
        self.cluster_size() / 2 + 1
    }
}

/// Statistics about the Raft node
#[derive(Debug, Clone, Default)]
pub struct RaftStats {
    /// Number of elections started
    pub elections_started: usize,
    /// Number of elections won
    pub elections_won: usize,
    /// Number of votes received
    pub votes_received: usize,
    /// Number of log entries replicated
    pub entries_replicated: usize,
    /// Number of heartbeats sent
    pub heartbeats_sent: usize,
    /// Number of terms seen
    pub terms_seen: usize,
}

/// A Raft consensus node
#[derive(Debug, Clone)]
pub struct RaftNode<T: Clone> {
    /// Configuration
    config: RaftConfig,
    /// Current state
    state: NodeState,
    /// Persistent state
    persistent: PersistentState<T>,
    /// Volatile state
    volatile: VolatileState,
    /// Leader state (only valid when leader)
    leader_state: Option<LeaderState>,
    /// Current leader (if known)
    current_leader: Option<NodeId>,
    /// Election timeout counter
    election_timeout: u64,
    /// Random election timeout target
    election_timeout_target: u64,
    /// Heartbeat counter (for leader)
    heartbeat_counter: u64,
    /// Votes received in current election
    votes_received: HashSet<NodeId>,
    /// Outgoing messages
    outbox: VecDeque<(NodeId, Message<T>)>,
    /// Applied commands (state machine output)
    applied_commands: Vec<T>,
    /// Statistics
    stats: RaftStats,
    /// Simple RNG state for election timeout
    rng_state: u64,
}

impl<T: Clone + std::fmt::Debug> RaftNode<T> {
    /// Create a new Raft node
    pub fn new(config: RaftConfig) -> Self {
        let election_timeout_target = config.election_timeout_min;
        Self {
            rng_state: config.id, // Seed with node ID
            config,
            state: NodeState::Follower,
            persistent: PersistentState::default(),
            volatile: VolatileState::default(),
            leader_state: None,
            current_leader: None,
            election_timeout: 0,
            election_timeout_target,
            heartbeat_counter: 0,
            votes_received: HashSet::new(),
            outbox: VecDeque::new(),
            applied_commands: Vec::new(),
            stats: RaftStats::default(),
        }
    }

    /// Get the node's ID
    pub fn id(&self) -> NodeId {
        self.config.id
    }

    /// Get the current state
    pub fn state(&self) -> NodeState {
        self.state
    }

    /// Get the current term
    pub fn current_term(&self) -> Term {
        self.persistent.current_term
    }

    /// Get the current leader (if known)
    pub fn current_leader(&self) -> Option<NodeId> {
        self.current_leader
    }

    /// Get the commit index
    pub fn commit_index(&self) -> LogIndex {
        self.volatile.commit_index
    }

    /// Get the log length
    pub fn log_len(&self) -> usize {
        self.persistent.log.len()
    }

    /// Get statistics
    pub fn stats(&self) -> &RaftStats {
        &self.stats
    }

    /// Get applied commands
    pub fn applied_commands(&self) -> &[T] {
        &self.applied_commands
    }

    /// Check if this node is the leader
    pub fn is_leader(&self) -> bool {
        self.state == NodeState::Leader
    }

    /// Get pending outgoing messages
    pub fn drain_messages(&mut self) -> Vec<(NodeId, Message<T>)> {
        self.outbox.drain(..).collect()
    }

    /// Simple pseudo-random number generator
    fn rand(&mut self) -> u64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.rng_state
    }

    /// Generate random election timeout
    fn randomize_election_timeout(&mut self) {
        let range = self.config.election_timeout_max - self.config.election_timeout_min;
        self.election_timeout_target =
            self.config.election_timeout_min + (self.rand() % (range + 1));
    }

    /// Get last log index
    fn last_log_index(&self) -> LogIndex {
        self.persistent.log.last().map(|e| e.index).unwrap_or(0)
    }

    /// Get last log term
    fn last_log_term(&self) -> Term {
        self.persistent.log.last().map(|e| e.term).unwrap_or(0)
    }

    /// Get log entry at index (1-based)
    fn log_entry(&self, index: LogIndex) -> Option<&LogEntry<T>> {
        if index == 0 || index as usize > self.persistent.log.len() {
            None
        } else {
            Some(&self.persistent.log[index as usize - 1])
        }
    }

    /// Get term at log index
    fn log_term(&self, index: LogIndex) -> Term {
        self.log_entry(index).map(|e| e.term).unwrap_or(0)
    }

    /// Process a tick (time unit)
    pub fn tick(&mut self) {
        match self.state {
            NodeState::Follower | NodeState::Candidate => {
                self.election_timeout += 1;
                if self.election_timeout >= self.election_timeout_target {
                    self.start_election();
                }
            }
            NodeState::Leader => {
                self.heartbeat_counter += 1;
                if self.heartbeat_counter >= self.config.heartbeat_interval {
                    self.send_heartbeats();
                    self.heartbeat_counter = 0;
                }
            }
        }
    }

    /// Start an election
    fn start_election(&mut self) {
        self.stats.elections_started += 1;
        self.persistent.current_term += 1;
        self.stats.terms_seen += 1;
        self.state = NodeState::Candidate;
        self.persistent.voted_for = Some(self.config.id);
        self.votes_received.clear();
        self.votes_received.insert(self.config.id);
        self.election_timeout = 0;
        self.randomize_election_timeout();

        // Send RequestVote to all peers
        let request = RequestVote {
            term: self.persistent.current_term,
            candidate_id: self.config.id,
            last_log_index: self.last_log_index(),
            last_log_term: self.last_log_term(),
        };

        for &peer in &self.config.peers {
            self.outbox
                .push_back((peer, Message::RequestVote(request.clone())));
        }
    }

    /// Send heartbeats to all peers (leader only)
    fn send_heartbeats(&mut self) {
        if self.state != NodeState::Leader {
            return;
        }

        self.stats.heartbeats_sent += 1;

        let peers: Vec<NodeId> = self.config.peers.clone();
        for peer in peers {
            self.send_append_entries(peer);
        }
    }

    /// Send AppendEntries to a specific peer
    fn send_append_entries(&mut self, peer: NodeId) {
        let leader_state = self.leader_state.as_ref().unwrap();
        let next_index = *leader_state.next_index.get(&peer).unwrap_or(&1);
        let prev_log_index = next_index.saturating_sub(1);
        let prev_log_term = self.log_term(prev_log_index);

        // Get entries to send
        let entries: Vec<LogEntry<T>> = self
            .persistent
            .log
            .iter()
            .filter(|e| e.index >= next_index)
            .cloned()
            .collect();

        let append = AppendEntries {
            term: self.persistent.current_term,
            leader_id: self.config.id,
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit: self.volatile.commit_index,
        };

        self.outbox
            .push_back((peer, Message::AppendEntries(append)));
    }

    /// Become leader
    fn become_leader(&mut self) {
        self.state = NodeState::Leader;
        self.current_leader = Some(self.config.id);
        self.leader_state = Some(LeaderState::new(self.last_log_index(), &self.config.peers));
        self.stats.elections_won += 1;

        // Send initial empty AppendEntries (heartbeat) to all peers
        self.send_heartbeats();
    }

    /// Step down to follower
    fn become_follower(&mut self, term: Term, leader: Option<NodeId>) {
        self.state = NodeState::Follower;
        self.persistent.current_term = term;
        self.persistent.voted_for = None;
        self.current_leader = leader;
        self.leader_state = None;
        self.election_timeout = 0;
        self.randomize_election_timeout();
    }

    /// Submit a command (only works if leader)
    pub fn submit_command(&mut self, command: T) -> Result<LogIndex, &'static str> {
        if self.state != NodeState::Leader {
            return Err("Not the leader");
        }

        let entry = LogEntry {
            term: self.persistent.current_term,
            index: self.last_log_index() + 1,
            command,
        };

        let index = entry.index;
        self.persistent.log.push(entry);

        // Immediately replicate to followers
        let peers: Vec<NodeId> = self.config.peers.clone();
        for peer in peers {
            self.send_append_entries(peer);
        }

        Ok(index)
    }

    /// Handle an incoming message
    pub fn handle_message(&mut self, from: NodeId, message: Message<T>) {
        match message {
            Message::RequestVote(rv) => self.handle_request_vote(from, rv),
            Message::RequestVoteResponse(rvr) => self.handle_request_vote_response(from, rvr),
            Message::AppendEntries(ae) => self.handle_append_entries(from, ae),
            Message::AppendEntriesResponse(aer) => self.handle_append_entries_response(from, aer),
        }
    }

    /// Handle RequestVote RPC
    fn handle_request_vote(&mut self, from: NodeId, rv: RequestVote) {
        // If term is outdated, reject
        if rv.term < self.persistent.current_term {
            self.outbox.push_back((
                from,
                Message::RequestVoteResponse(RequestVoteResponse {
                    term: self.persistent.current_term,
                    vote_granted: false,
                }),
            ));
            return;
        }

        // If newer term, become follower
        if rv.term > self.persistent.current_term {
            self.become_follower(rv.term, None);
        }

        // Check if we can vote for this candidate
        let can_vote = self.persistent.voted_for.is_none()
            || self.persistent.voted_for == Some(rv.candidate_id);

        // Check if candidate's log is at least as up-to-date
        let log_ok = rv.last_log_term > self.last_log_term()
            || (rv.last_log_term == self.last_log_term()
                && rv.last_log_index >= self.last_log_index());

        let vote_granted = can_vote && log_ok;

        if vote_granted {
            self.persistent.voted_for = Some(rv.candidate_id);
            self.election_timeout = 0;
            self.randomize_election_timeout();
        }

        self.outbox.push_back((
            from,
            Message::RequestVoteResponse(RequestVoteResponse {
                term: self.persistent.current_term,
                vote_granted,
            }),
        ));
    }

    /// Handle RequestVote response
    fn handle_request_vote_response(&mut self, from: NodeId, rvr: RequestVoteResponse) {
        if rvr.term > self.persistent.current_term {
            self.become_follower(rvr.term, None);
            return;
        }

        if self.state != NodeState::Candidate {
            return;
        }

        if rvr.term != self.persistent.current_term {
            return;
        }

        if rvr.vote_granted {
            self.votes_received.insert(from);
            self.stats.votes_received += 1;

            // Check if we have majority
            if self.votes_received.len() >= self.config.majority() {
                self.become_leader();
            }
        }
    }

    /// Handle AppendEntries RPC
    fn handle_append_entries(&mut self, from: NodeId, ae: AppendEntries<T>) {
        // Reply false if term < currentTerm
        if ae.term < self.persistent.current_term {
            self.outbox.push_back((
                from,
                Message::AppendEntriesResponse(AppendEntriesResponse {
                    term: self.persistent.current_term,
                    success: false,
                    match_index: 0,
                }),
            ));
            return;
        }

        // If newer term or we're candidate and see leader, become follower
        if ae.term > self.persistent.current_term || self.state == NodeState::Candidate {
            self.become_follower(ae.term, Some(ae.leader_id));
        } else {
            // Reset election timeout on valid AppendEntries
            self.election_timeout = 0;
            self.current_leader = Some(ae.leader_id);
        }

        // Reply false if log doesn't contain an entry at prevLogIndex
        // whose term matches prevLogTerm
        if ae.prev_log_index > 0 {
            let log_term = self.log_term(ae.prev_log_index);
            if log_term != ae.prev_log_term {
                self.outbox.push_back((
                    from,
                    Message::AppendEntriesResponse(AppendEntriesResponse {
                        term: self.persistent.current_term,
                        success: false,
                        match_index: self.last_log_index(),
                    }),
                ));
                return;
            }
        }

        // Process entries
        for entry in ae.entries {
            // If existing entry conflicts with new one, delete it and all following
            if entry.index as usize <= self.persistent.log.len() {
                let existing_term = self.log_term(entry.index);
                if existing_term != entry.term {
                    self.persistent.log.truncate(entry.index as usize - 1);
                }
            }

            // Append new entry if not already present
            if entry.index as usize > self.persistent.log.len() {
                self.persistent.log.push(entry);
                self.stats.entries_replicated += 1;
            }
        }

        // Update commit index
        if ae.leader_commit > self.volatile.commit_index {
            self.volatile.commit_index = ae.leader_commit.min(self.last_log_index());
            self.apply_committed_entries();
        }

        self.outbox.push_back((
            from,
            Message::AppendEntriesResponse(AppendEntriesResponse {
                term: self.persistent.current_term,
                success: true,
                match_index: self.last_log_index(),
            }),
        ));
    }

    /// Handle AppendEntries response
    fn handle_append_entries_response(&mut self, from: NodeId, aer: AppendEntriesResponse) {
        if aer.term > self.persistent.current_term {
            self.become_follower(aer.term, None);
            return;
        }

        if self.state != NodeState::Leader {
            return;
        }

        let leader_state = self.leader_state.as_mut().unwrap();

        if aer.success {
            // Update match_index and next_index
            leader_state.match_index.insert(from, aer.match_index);
            leader_state.next_index.insert(from, aer.match_index + 1);

            // Try to advance commit index
            self.try_advance_commit_index();
        } else {
            // Decrement next_index and retry
            let next = leader_state.next_index.entry(from).or_insert(1);
            *next = next.saturating_sub(1).max(1);
            self.send_append_entries(from);
        }
    }

    /// Try to advance the commit index (leader only)
    fn try_advance_commit_index(&mut self) {
        if self.state != NodeState::Leader {
            return;
        }

        let leader_state = self.leader_state.as_ref().unwrap();

        // Find the highest index N such that:
        // - N > commitIndex
        // - log[N].term == currentTerm
        // - majority of matchIndex[i] >= N
        for n in (self.volatile.commit_index + 1)..=self.last_log_index() {
            if self.log_term(n) != self.persistent.current_term {
                continue;
            }

            let mut count = 1; // Count self
            for &peer in &self.config.peers {
                if let Some(&match_idx) = leader_state.match_index.get(&peer)
                    && match_idx >= n
                {
                    count += 1;
                }
            }

            if count >= self.config.majority() {
                self.volatile.commit_index = n;
            }
        }

        self.apply_committed_entries();
    }

    /// Apply committed entries to state machine
    fn apply_committed_entries(&mut self) {
        while self.volatile.last_applied < self.volatile.commit_index {
            self.volatile.last_applied += 1;
            if let Some(entry) = self.log_entry(self.volatile.last_applied) {
                self.applied_commands.push(entry.command.clone());
            }
        }
    }
}

/// A simple cluster simulation for testing
#[derive(Debug)]
pub struct RaftCluster<T: Clone + std::fmt::Debug> {
    /// All nodes in the cluster
    pub nodes: HashMap<NodeId, RaftNode<T>>,
    /// Message queue
    messages: VecDeque<(NodeId, NodeId, Message<T>)>,
    /// Total ticks elapsed
    pub ticks: u64,
}

impl<T: Clone + std::fmt::Debug> RaftCluster<T> {
    /// Create a new cluster with the given number of nodes
    pub fn new(num_nodes: usize) -> Self {
        let node_ids: Vec<NodeId> = (1..=num_nodes as NodeId).collect();
        let mut nodes = HashMap::new();

        for &id in &node_ids {
            let peers: Vec<NodeId> = node_ids.iter().copied().filter(|&n| n != id).collect();
            let config = RaftConfig::new(id, peers);
            nodes.insert(id, RaftNode::new(config));
        }

        Self {
            nodes,
            messages: VecDeque::new(),
            ticks: 0,
        }
    }

    /// Run a single tick on all nodes
    pub fn tick(&mut self) {
        self.ticks += 1;

        // Tick all nodes
        for node in self.nodes.values_mut() {
            node.tick();
        }

        // Collect outgoing messages
        for node in self.nodes.values_mut() {
            let msgs = node.drain_messages();
            for (to, msg) in msgs {
                self.messages.push_back((node.id(), to, msg));
            }
        }

        // Deliver messages
        while let Some((from, to, msg)) = self.messages.pop_front() {
            if let Some(node) = self.nodes.get_mut(&to) {
                node.handle_message(from, msg);
            }
        }

        // Collect any response messages
        for node in self.nodes.values_mut() {
            let msgs = node.drain_messages();
            for (to, msg) in msgs {
                self.messages.push_back((node.id(), to, msg));
            }
        }
    }

    /// Run until a leader is elected or max ticks reached
    pub fn elect_leader(&mut self, max_ticks: u64) -> Option<NodeId> {
        for _ in 0..max_ticks {
            self.tick();
            if let Some(leader) = self.get_leader() {
                return Some(leader);
            }
        }
        None
    }

    /// Get current leader (if any)
    pub fn get_leader(&self) -> Option<NodeId> {
        self.nodes.values().find(|n| n.is_leader()).map(|n| n.id())
    }

    /// Submit a command to the cluster (finds leader automatically)
    pub fn submit(&mut self, command: T) -> Result<LogIndex, &'static str> {
        let leader_id = self.get_leader().ok_or("No leader")?;
        self.nodes
            .get_mut(&leader_id)
            .unwrap()
            .submit_command(command)
    }

    /// Run until command is committed or max ticks reached
    pub fn replicate(&mut self, max_ticks: u64) -> bool {
        let leader_id = match self.get_leader() {
            Some(id) => id,
            None => return false,
        };

        let target_commit = self.nodes.get(&leader_id).unwrap().last_log_index();

        for _ in 0..max_ticks {
            self.tick();

            // Check if majority have committed
            let committed_count = self
                .nodes
                .values()
                .filter(|n| n.commit_index() >= target_commit)
                .count();

            if committed_count > self.nodes.len() / 2 {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let config = RaftConfig::new(1, vec![2, 3]);
        let node: RaftNode<String> = RaftNode::new(config);

        assert_eq!(node.id(), 1);
        assert_eq!(node.state(), NodeState::Follower);
        assert_eq!(node.current_term(), 0);
        assert_eq!(node.log_len(), 0);
        assert!(!node.is_leader());
    }

    #[test]
    fn test_cluster_creation() {
        let cluster: RaftCluster<String> = RaftCluster::new(3);

        assert_eq!(cluster.nodes.len(), 3);
        assert!(cluster.get_leader().is_none());
    }

    #[test]
    fn test_leader_election() {
        let mut cluster: RaftCluster<String> = RaftCluster::new(3);

        let leader = cluster.elect_leader(500);
        assert!(leader.is_some());

        let leader_id = leader.unwrap();
        assert!(cluster.nodes.get(&leader_id).unwrap().is_leader());

        // Run a few more ticks to ensure heartbeats are sent and processed
        for _ in 0..100 {
            cluster.tick();
        }

        // All nodes should know the leader
        for node in cluster.nodes.values() {
            if node.id() != leader_id {
                assert_eq!(node.current_leader(), Some(leader_id));
            }
        }
    }

    #[test]
    fn test_log_replication() {
        let mut cluster: RaftCluster<String> = RaftCluster::new(3);

        // Elect leader
        cluster.elect_leader(500);
        let _leader_id = cluster.get_leader().unwrap();

        // Submit a command
        let result = cluster.submit("command1".to_string());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);

        // Replicate
        assert!(cluster.replicate(100));

        // All nodes should have the entry committed
        for node in cluster.nodes.values() {
            assert!(node.commit_index() >= 1);
            assert_eq!(node.applied_commands().len(), 1);
            assert_eq!(node.applied_commands()[0], "command1");
        }
    }

    #[test]
    fn test_multiple_commands() {
        let mut cluster: RaftCluster<i32> = RaftCluster::new(5);

        cluster.elect_leader(500);

        // Submit multiple commands
        for i in 1..=5 {
            assert!(cluster.submit(i * 10).is_ok());
            assert!(cluster.replicate(100));
        }

        let leader_id = cluster.get_leader().unwrap();
        let leader = cluster.nodes.get(&leader_id).unwrap();

        assert_eq!(leader.log_len(), 5);
        assert_eq!(leader.applied_commands(), &[10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_term_advancement() {
        let mut cluster: RaftCluster<String> = RaftCluster::new(3);

        cluster.elect_leader(500);
        let initial_term = cluster.nodes.values().next().unwrap().current_term();

        // All nodes should be on the same term
        for node in cluster.nodes.values() {
            assert_eq!(node.current_term(), initial_term);
        }

        assert!(initial_term > 0);
    }

    #[test]
    fn test_submit_to_non_leader() {
        let config = RaftConfig::new(1, vec![2, 3]);
        let mut node: RaftNode<String> = RaftNode::new(config);

        // Node is follower, should reject
        let result = node.submit_command("test".to_string());
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Not the leader");
    }

    #[test]
    fn test_config() {
        let config = RaftConfig::new(1, vec![2, 3, 4, 5]);

        assert_eq!(config.cluster_size(), 5);
        assert_eq!(config.majority(), 3);

        let config3 = RaftConfig::new(1, vec![2, 3]);
        assert_eq!(config3.cluster_size(), 3);
        assert_eq!(config3.majority(), 2);
    }

    #[test]
    fn test_election_timeout() {
        let config = RaftConfig::new(1, vec![2, 3]);
        let mut node: RaftNode<String> = RaftNode::new(config);

        // Tick until election starts
        for _ in 0..500 {
            node.tick();
            if node.state() == NodeState::Candidate {
                break;
            }
        }

        assert_eq!(node.state(), NodeState::Candidate);
        assert!(node.current_term() > 0);
    }

    #[test]
    fn test_request_vote() {
        let config1 = RaftConfig::new(1, vec![2, 3]);
        let config2 = RaftConfig::new(2, vec![1, 3]);

        let mut node1: RaftNode<String> = RaftNode::new(config1);
        let mut node2: RaftNode<String> = RaftNode::new(config2);

        // Node 1 starts election
        for _ in 0..500 {
            node1.tick();
            if node1.state() == NodeState::Candidate {
                break;
            }
        }

        // Get request vote message
        let messages = node1.drain_messages();
        let rv_msg = messages
            .into_iter()
            .find(|(to, _)| *to == 2)
            .map(|(_, m)| m);

        if let Some(Message::RequestVote(rv)) = rv_msg {
            node2.handle_message(1, Message::RequestVote(rv));

            // Node 2 should have voted
            let responses = node2.drain_messages();
            assert!(!responses.is_empty());

            if let Some((_, Message::RequestVoteResponse(rvr))) = responses.into_iter().next() {
                assert!(rvr.vote_granted);
            }
        }
    }

    #[test]
    fn test_stats() {
        let mut cluster: RaftCluster<String> = RaftCluster::new(3);

        cluster.elect_leader(500);
        cluster.submit("test".to_string()).unwrap();
        cluster.replicate(100);

        let leader_id = cluster.get_leader().unwrap();
        let leader = cluster.nodes.get(&leader_id).unwrap();

        assert!(leader.stats().elections_won >= 1);
        assert!(leader.stats().terms_seen >= 1);
    }

    #[test]
    fn test_five_node_cluster() {
        let mut cluster: RaftCluster<String> = RaftCluster::new(5);

        let leader = cluster.elect_leader(500);
        assert!(leader.is_some());

        // Submit and replicate
        cluster.submit("hello".to_string()).unwrap();
        assert!(cluster.replicate(100));

        // All nodes should have committed
        let committed_count = cluster
            .nodes
            .values()
            .filter(|n| n.commit_index() >= 1)
            .count();
        assert!(committed_count >= 3); // Majority
    }

    #[test]
    fn test_append_entries_heartbeat() {
        let config = RaftConfig::new(1, vec![2]);
        let mut leader: RaftNode<String> = RaftNode::new(config);

        // Manually become leader
        leader.persistent.current_term = 1;
        leader.state = NodeState::Leader;
        leader.leader_state = Some(LeaderState::new(0, &[2]));

        leader.send_heartbeats();

        let messages = leader.drain_messages();
        assert_eq!(messages.len(), 1);

        if let (2, Message::AppendEntries(ae)) = &messages[0] {
            assert_eq!(ae.term, 1);
            assert_eq!(ae.leader_id, 1);
            assert!(ae.entries.is_empty()); // Heartbeat has no entries
        } else {
            panic!("Expected AppendEntries message");
        }
    }

    #[test]
    fn test_log_entry() {
        let mut cluster: RaftCluster<i32> = RaftCluster::new(3);
        cluster.elect_leader(500);

        cluster.submit(42).unwrap();
        cluster.replicate(100);

        let leader_id = cluster.get_leader().unwrap();
        let leader = cluster.nodes.get(&leader_id).unwrap();

        assert_eq!(leader.log_len(), 1);

        let entry = leader.log_entry(1).unwrap();
        assert_eq!(entry.command, 42);
        assert_eq!(entry.index, 1);
        assert!(entry.term > 0);
    }

    #[test]
    fn test_follower_log_consistency() {
        let mut cluster: RaftCluster<String> = RaftCluster::new(3);
        cluster.elect_leader(500);

        // Submit multiple commands
        for i in 0..5 {
            cluster.submit(format!("cmd{}", i)).unwrap();
            cluster.replicate(100);
        }

        // All followers should have same log as leader
        let leader_id = cluster.get_leader().unwrap();
        let leader_log_len = cluster.nodes.get(&leader_id).unwrap().log_len();

        for node in cluster.nodes.values() {
            if node.id() != leader_id {
                assert_eq!(node.log_len(), leader_log_len);
            }
        }
    }

    #[test]
    fn test_node_state_transitions() {
        let config = RaftConfig::new(1, vec![2, 3]);
        let mut node: RaftNode<String> = RaftNode::new(config);

        assert_eq!(node.state(), NodeState::Follower);

        // Trigger election
        for _ in 0..500 {
            node.tick();
            if node.state() == NodeState::Candidate {
                break;
            }
        }
        assert_eq!(node.state(), NodeState::Candidate);
    }
}
