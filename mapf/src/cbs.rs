// Conflict-Based Search (CBS)
//
// Optimal multi-agent pathfinding algorithm. Finds collision-free paths for
// multiple agents from their starts to goals on a shared graph.
//
// Two-level search:
// - High level: Search over conflict tree (CT) nodes, each representing a set of constraints
// - Low level: Single-agent A* respecting constraints from the CT node
//
// Key concepts:
// - Constraint: (agent, location, time) or (agent, edge, time) - agent can't be there/then
// - Conflict: Two agents at same location/edge at same time
// - CT Node: Set of constraints + paths for all agents + total cost
//
// Optimal because it explores CT nodes in cost order and only adds necessary constraints.
//
// Time: Exponential worst case, but often efficient in practice
// Space: O(agents * path_length * CT_nodes)

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

pub type AgentId = usize;
pub type Time = u32;
pub type Location = (i32, i32);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Constraint {
    Vertex(AgentId, Location, Time),
    Edge(AgentId, Location, Location, Time), // can't move from loc1 to loc2 at time t
}

#[derive(Clone, Debug)]
pub struct Conflict {
    pub agent1: AgentId,
    pub agent2: AgentId,
    pub location: Location,
    pub time: Time,
    pub is_edge: bool,
    pub location2: Option<Location>, // for edge conflicts
}

// Map definition
pub struct GridMap {
    pub width: i32,
    pub height: i32,
    pub obstacles: HashSet<Location>,
}

impl GridMap {
    pub fn new(width: i32, height: i32) -> Self {
        Self {
            width,
            height,
            obstacles: HashSet::new(),
        }
    }

    pub fn with_obstacles(mut self, obstacles: Vec<Location>) -> Self {
        self.obstacles = obstacles.into_iter().collect();
        self
    }

    fn is_valid(&self, loc: Location) -> bool {
        loc.0 >= 0
            && loc.0 < self.width
            && loc.1 >= 0
            && loc.1 < self.height
            && !self.obstacles.contains(&loc)
    }

    fn neighbors(&self, loc: Location) -> Vec<Location> {
        let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]; // include wait
        dirs.iter()
            .map(|(dx, dy)| (loc.0 + dx, loc.1 + dy))
            .filter(|&l| self.is_valid(l))
            .collect()
    }
}

// Agent definition
#[derive(Clone, Debug)]
pub struct Agent {
    pub start: Location,
    pub goal: Location,
}

// Path for a single agent
pub type Path = Vec<Location>;

// CT Node for high-level search
#[derive(Clone)]
struct CTNode {
    constraints: Vec<Constraint>,
    paths: Vec<Path>,
    cost: u32,
}

impl PartialEq for CTNode {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for CTNode {}

impl PartialOrd for CTNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CTNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost) // min-heap
    }
}

// Result of CBS
#[derive(Debug, Clone)]
pub struct CBSResult {
    pub paths: Vec<Path>,
    pub total_cost: u32,
    pub ct_nodes_expanded: u32,
}

// Main CBS function
pub fn cbs(map: &GridMap, agents: &[Agent], max_time: Time) -> Option<CBSResult> {
    let mut open: BinaryHeap<CTNode> = BinaryHeap::new();
    let mut ct_nodes_expanded = 0;

    // Initial CT node: no constraints, compute initial paths
    let initial_paths: Vec<Path> = agents
        .iter()
        .enumerate()
        .map(|(id, agent)| low_level_search(map, agent, &[], id, max_time))
        .collect::<Option<Vec<_>>>()?;

    let initial_cost = initial_paths.iter().map(|p| p.len() as u32 - 1).sum();

    open.push(CTNode {
        constraints: vec![],
        paths: initial_paths,
        cost: initial_cost,
    });

    while let Some(node) = open.pop() {
        ct_nodes_expanded += 1;

        // Find first conflict
        if let Some(conflict) = find_conflict(&node.paths) {
            // Branch: create two children with new constraints
            for (agent, constraint) in resolve_conflict(&conflict) {
                let mut new_constraints = node.constraints.clone();
                new_constraints.push(constraint);

                // Replan for the constrained agent
                let agent_constraints: Vec<_> = new_constraints
                    .iter()
                    .filter(|c| match c {
                        Constraint::Vertex(a, _, _) => *a == agent,
                        Constraint::Edge(a, _, _, _) => *a == agent,
                    })
                    .cloned()
                    .collect();

                if let Some(new_path) =
                    low_level_search(map, &agents[agent], &agent_constraints, agent, max_time)
                {
                    let mut new_paths = node.paths.clone();
                    new_paths[agent] = new_path;

                    let new_cost = new_paths.iter().map(|p| p.len() as u32 - 1).sum();

                    open.push(CTNode {
                        constraints: new_constraints,
                        paths: new_paths,
                        cost: new_cost,
                    });
                }
            }
        } else {
            // No conflicts - solution found
            return Some(CBSResult {
                paths: node.paths,
                total_cost: node.cost,
                ct_nodes_expanded,
            });
        }
    }

    None
}

// Find first conflict between any two agents
fn find_conflict(paths: &[Path]) -> Option<Conflict> {
    let max_len = paths.iter().map(|p| p.len()).max().unwrap_or(0);

    for t in 0..max_len {
        // Check vertex conflicts
        for i in 0..paths.len() {
            for j in (i + 1)..paths.len() {
                let loc_i = get_location(&paths[i], t);
                let loc_j = get_location(&paths[j], t);

                if loc_i == loc_j {
                    return Some(Conflict {
                        agent1: i,
                        agent2: j,
                        location: loc_i,
                        time: t as Time,
                        is_edge: false,
                        location2: None,
                    });
                }

                // Check edge conflicts (agents swapping positions)
                if t > 0 {
                    let prev_i = get_location(&paths[i], t - 1);
                    let prev_j = get_location(&paths[j], t - 1);

                    if loc_i == prev_j && loc_j == prev_i {
                        return Some(Conflict {
                            agent1: i,
                            agent2: j,
                            location: prev_i,
                            time: t as Time,
                            is_edge: true,
                            location2: Some(loc_i),
                        });
                    }
                }
            }
        }
    }

    None
}

fn get_location(path: &Path, time: usize) -> Location {
    if time < path.len() {
        path[time]
    } else {
        *path.last().unwrap() // agent stays at goal
    }
}

// Generate constraints to resolve a conflict
fn resolve_conflict(conflict: &Conflict) -> Vec<(AgentId, Constraint)> {
    if conflict.is_edge {
        let loc2 = conflict.location2.unwrap();
        vec![
            (
                conflict.agent1,
                Constraint::Edge(conflict.agent1, conflict.location, loc2, conflict.time),
            ),
            (
                conflict.agent2,
                Constraint::Edge(conflict.agent2, loc2, conflict.location, conflict.time),
            ),
        ]
    } else {
        vec![
            (
                conflict.agent1,
                Constraint::Vertex(conflict.agent1, conflict.location, conflict.time),
            ),
            (
                conflict.agent2,
                Constraint::Vertex(conflict.agent2, conflict.location, conflict.time),
            ),
        ]
    }
}

// Low-level A* search for single agent with constraints
fn low_level_search(
    map: &GridMap,
    agent: &Agent,
    constraints: &[Constraint],
    agent_id: AgentId,
    max_time: Time,
) -> Option<Path> {
    #[derive(Clone, Eq, PartialEq)]
    struct State {
        loc: Location,
        time: Time,
        g: u32,
        f: u32,
        path: Vec<Location>,
    }

    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other.f.cmp(&self.f).then_with(|| other.g.cmp(&self.g))
        }
    }

    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let heuristic =
        |loc: Location| ((loc.0 - agent.goal.0).abs() + (loc.1 - agent.goal.1).abs()) as u32;

    let mut open: BinaryHeap<State> = BinaryHeap::new();
    let mut closed: HashSet<(Location, Time)> = HashSet::new();

    let h = heuristic(agent.start);
    open.push(State {
        loc: agent.start,
        time: 0,
        g: 0,
        f: h,
        path: vec![agent.start],
    });

    // Build constraint lookup
    let vertex_constraints: HashSet<(Location, Time)> = constraints
        .iter()
        .filter_map(|c| match c {
            Constraint::Vertex(a, loc, t) if *a == agent_id => Some((*loc, *t)),
            _ => None,
        })
        .collect();

    let edge_constraints: HashSet<(Location, Location, Time)> = constraints
        .iter()
        .filter_map(|c| match c {
            Constraint::Edge(a, l1, l2, t) if *a == agent_id => Some((*l1, *l2, *t)),
            _ => None,
        })
        .collect();

    while let Some(current) = open.pop() {
        if current.loc == agent.goal {
            return Some(current.path);
        }

        if current.time >= max_time {
            continue;
        }

        if closed.contains(&(current.loc, current.time)) {
            continue;
        }
        closed.insert((current.loc, current.time));

        let next_time = current.time + 1;

        for next_loc in map.neighbors(current.loc) {
            // Check vertex constraint
            if vertex_constraints.contains(&(next_loc, next_time)) {
                continue;
            }

            // Check edge constraint
            if edge_constraints.contains(&(current.loc, next_loc, next_time)) {
                continue;
            }

            if closed.contains(&(next_loc, next_time)) {
                continue;
            }

            let g = current.g + 1;
            let h = heuristic(next_loc);
            let mut new_path = current.path.clone();
            new_path.push(next_loc);

            open.push(State {
                loc: next_loc,
                time: next_time,
                g,
                f: g + h,
                path: new_path,
            });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cbs_no_conflict() {
        let map = GridMap::new(5, 5);
        let agents = vec![
            Agent {
                start: (0, 0),
                goal: (4, 0),
            },
            Agent {
                start: (0, 4),
                goal: (4, 4),
            },
        ];

        let result = cbs(&map, &agents, 20).unwrap();

        assert_eq!(result.paths.len(), 2);
        assert_eq!(result.paths[0].first(), Some(&(0, 0)));
        assert_eq!(result.paths[0].last(), Some(&(4, 0)));
        assert_eq!(result.paths[1].first(), Some(&(0, 4)));
        assert_eq!(result.paths[1].last(), Some(&(4, 4)));
    }

    #[test]
    fn test_cbs_head_on_conflict() {
        // Two agents need to pass each other - use 2D grid so they can maneuver
        let map = GridMap::new(5, 3);
        let agents = vec![
            Agent {
                start: (0, 1),
                goal: (4, 1),
            },
            Agent {
                start: (4, 1),
                goal: (0, 1),
            },
        ];

        let result = cbs(&map, &agents, 20).unwrap();

        // Both agents reach goals
        assert_eq!(result.paths[0].last(), Some(&(4, 1)));
        assert_eq!(result.paths[1].last(), Some(&(0, 1)));

        // Verify no conflicts in solution
        let max_len = result.paths.iter().map(|p| p.len()).max().unwrap();
        for t in 0..max_len {
            let loc0 = get_location(&result.paths[0], t);
            let loc1 = get_location(&result.paths[1], t);
            assert_ne!(loc0, loc1, "Vertex conflict at time {}", t);
        }
    }

    #[test]
    fn test_cbs_vertex_conflict() {
        // Agents cross at same point
        let map = GridMap::new(3, 3);
        let agents = vec![
            Agent {
                start: (0, 1),
                goal: (2, 1),
            },
            Agent {
                start: (1, 0),
                goal: (1, 2),
            },
        ];

        let result = cbs(&map, &agents, 20).unwrap();

        // Verify no conflicts in solution
        let paths = &result.paths;
        let max_len = paths.iter().map(|p| p.len()).max().unwrap();

        for t in 0..max_len {
            let loc0 = get_location(&paths[0], t);
            let loc1 = get_location(&paths[1], t);
            assert_ne!(loc0, loc1, "Conflict at time {}", t);
        }
    }

    #[test]
    fn test_cbs_with_obstacles() {
        let map = GridMap::new(5, 5).with_obstacles(vec![(2, 2), (2, 1), (2, 3)]);
        let agents = vec![
            Agent {
                start: (0, 2),
                goal: (4, 2),
            },
            Agent {
                start: (1, 0),
                goal: (1, 4),
            },
        ];

        let result = cbs(&map, &agents, 30).unwrap();

        // Both agents should reach their goals
        assert_eq!(result.paths[0].last(), Some(&(4, 2)));
        assert_eq!(result.paths[1].last(), Some(&(1, 4)));
    }

    #[test]
    fn test_cbs_single_agent() {
        let map = GridMap::new(5, 5);
        let agents = vec![Agent {
            start: (0, 0),
            goal: (2, 2),
        }];

        let result = cbs(&map, &agents, 20).unwrap();

        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].first(), Some(&(0, 0)));
        assert_eq!(result.paths[0].last(), Some(&(2, 2)));
        assert_eq!(result.total_cost, 4); // Manhattan distance
    }

    #[test]
    fn test_cbs_three_agents() {
        let map = GridMap::new(5, 5);
        let agents = vec![
            Agent {
                start: (0, 0),
                goal: (4, 4),
            },
            Agent {
                start: (4, 0),
                goal: (0, 4),
            },
            Agent {
                start: (2, 0),
                goal: (2, 4),
            },
        ];

        let result = cbs(&map, &agents, 30).unwrap();

        assert_eq!(result.paths.len(), 3);
        // All agents reach goals
        assert_eq!(result.paths[0].last(), Some(&(4, 4)));
        assert_eq!(result.paths[1].last(), Some(&(0, 4)));
        assert_eq!(result.paths[2].last(), Some(&(2, 4)));
    }
}
