// Priority-Based Planning (Prioritized Planning)
//
// Simple and fast multi-agent pathfinding algorithm. Agents plan sequentially
// based on priority - higher priority agents plan first, lower priority agents
// treat higher priority paths as moving obstacles.
//
// Algorithm:
// 1. Order agents by priority (can be arbitrary or based on some heuristic)
// 2. For each agent in priority order:
//    - Plan path using space-time A* avoiding all higher-priority agent paths
//    - Add this path to the set of constraints for lower-priority agents
//
// Key differences from CBS:
// - Not optimal: solution quality depends on priority ordering
// - Much faster: O(n) single-agent searches vs exponential constraint tree
// - Complete for most cases, but can fail if low-priority agent is boxed in
//
// Time: O(n * single_agent_search) where n = number of agents
// Space: O(n * path_length)
//
// Use when: Speed matters more than optimality, or as initial solution for CBS

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

pub type AgentId = usize;
pub type Time = u32;
pub type Location = (i32, i32);

// Reuse GridMap and Agent from CBS or define locally for independence
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

#[derive(Clone, Debug)]
pub struct Agent {
    pub start: Location,
    pub goal: Location,
}

pub type Path = Vec<Location>;

#[derive(Debug, Clone)]
pub struct PriorityPlanningResult {
    pub paths: Vec<Path>,
    pub total_cost: u32,
    pub planning_order: Vec<AgentId>, // order in which agents were planned
}

// Priority ordering strategies
#[derive(Clone, Copy, Debug)]
pub enum PriorityStrategy {
    // Agents with longer shortest paths plan first (they need more flexibility)
    LongestPathFirst,
    // Agents with shorter shortest paths plan first
    ShortestPathFirst,
    // Use the order given in the input
    InputOrder,
    // Agents closer to center plan first (more likely to conflict)
    CenterFirst,
}

// Main priority-based planning function
pub fn priority_planning(
    map: &GridMap,
    agents: &[Agent],
    max_time: Time,
    strategy: PriorityStrategy,
) -> Option<PriorityPlanningResult> {
    if agents.is_empty() {
        return Some(PriorityPlanningResult {
            paths: vec![],
            total_cost: 0,
            planning_order: vec![],
        });
    }

    // Determine planning order based on strategy
    let planning_order = compute_priority_order(map, agents, strategy);

    // Constraints from higher-priority agents: (location, time) pairs to avoid
    let mut reserved_vertices: HashSet<(Location, Time)> = HashSet::new();
    let mut reserved_edges: HashSet<(Location, Location, Time)> = HashSet::new();

    let mut paths: Vec<Option<Path>> = vec![None; agents.len()];
    let mut total_cost = 0;

    // Plan for each agent in priority order
    for &agent_id in &planning_order {
        let agent = &agents[agent_id];

        // Plan path avoiding reserved locations
        let path = space_time_astar(map, agent, &reserved_vertices, &reserved_edges, max_time)?;

        // Reserve this path for lower-priority agents
        reserve_path(&path, &mut reserved_vertices, &mut reserved_edges);

        total_cost += path.len() as u32 - 1;
        paths[agent_id] = Some(path);
    }

    // Unwrap all paths (they should all be Some at this point)
    let paths: Vec<Path> = paths.into_iter().map(|p| p.unwrap()).collect();

    Some(PriorityPlanningResult {
        paths,
        total_cost,
        planning_order,
    })
}

// Compute priority order based on strategy
fn compute_priority_order(
    map: &GridMap,
    agents: &[Agent],
    strategy: PriorityStrategy,
) -> Vec<AgentId> {
    let mut order: Vec<AgentId> = (0..agents.len()).collect();

    match strategy {
        PriorityStrategy::InputOrder => {
            // Keep original order
        }
        PriorityStrategy::LongestPathFirst => {
            // Sort by Manhattan distance, longest first
            order.sort_by(|&a, &b| {
                let dist_a = manhattan_distance(agents[a].start, agents[a].goal);
                let dist_b = manhattan_distance(agents[b].start, agents[b].goal);
                dist_b.cmp(&dist_a) // Reverse for longest first
            });
        }
        PriorityStrategy::ShortestPathFirst => {
            order.sort_by(|&a, &b| {
                let dist_a = manhattan_distance(agents[a].start, agents[a].goal);
                let dist_b = manhattan_distance(agents[b].start, agents[b].goal);
                dist_a.cmp(&dist_b)
            });
        }
        PriorityStrategy::CenterFirst => {
            let center = (map.width / 2, map.height / 2);
            order.sort_by(|&a, &b| {
                let dist_a = manhattan_distance(agents[a].start, center)
                    + manhattan_distance(agents[a].goal, center);
                let dist_b = manhattan_distance(agents[b].start, center)
                    + manhattan_distance(agents[b].goal, center);
                dist_a.cmp(&dist_b) // Closer to center first
            });
        }
    }

    order
}

fn manhattan_distance(a: Location, b: Location) -> i32 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}

// Reserve all positions along a path
fn reserve_path(
    path: &Path,
    reserved_vertices: &mut HashSet<(Location, Time)>,
    reserved_edges: &mut HashSet<(Location, Location, Time)>,
) {
    for (t, loc) in path.iter().enumerate() {
        reserved_vertices.insert((*loc, t as Time));

        // Also reserve edges for swap prevention
        if t > 0 {
            let prev = path[t - 1];
            // Reverse edge: if agent moves from prev to loc at time t,
            // other agents can't move from loc to prev at time t
            reserved_edges.insert((*loc, prev, t as Time));
        }
    }

    // Reserve goal position for all future time steps (agent stays at goal)
    if let Some(&goal) = path.last() {
        let final_time = path.len() as Time;
        // Reserve for reasonable future horizon (matching max_time typically)
        for t in final_time..(final_time + 50) {
            reserved_vertices.insert((goal, t));
        }
    }
}

// Space-time A* search avoiding reserved locations
fn space_time_astar(
    map: &GridMap,
    agent: &Agent,
    reserved_vertices: &HashSet<(Location, Time)>,
    reserved_edges: &HashSet<(Location, Location, Time)>,
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
            // Check vertex reservation
            if reserved_vertices.contains(&(next_loc, next_time)) {
                continue;
            }

            // Check edge reservation (swap prevention)
            if reserved_edges.contains(&(current.loc, next_loc, next_time)) {
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

// Convenience function with default strategy
pub fn priority_planning_default(
    map: &GridMap,
    agents: &[Agent],
    max_time: Time,
) -> Option<PriorityPlanningResult> {
    priority_planning(map, agents, max_time, PriorityStrategy::LongestPathFirst)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_location(path: &Path, time: usize) -> Location {
        if time < path.len() {
            path[time]
        } else {
            *path.last().unwrap()
        }
    }

    fn verify_no_conflicts(paths: &[Path]) -> bool {
        let max_len = paths.iter().map(|p| p.len()).max().unwrap_or(0);

        for t in 0..max_len {
            for i in 0..paths.len() {
                for j in (i + 1)..paths.len() {
                    let loc_i = get_location(&paths[i], t);
                    let loc_j = get_location(&paths[j], t);

                    // Vertex conflict
                    if loc_i == loc_j {
                        return false;
                    }

                    // Edge conflict (swap)
                    if t > 0 {
                        let prev_i = get_location(&paths[i], t - 1);
                        let prev_j = get_location(&paths[j], t - 1);
                        if loc_i == prev_j && loc_j == prev_i {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    #[test]
    fn test_priority_planning_no_conflict() {
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

        let result = priority_planning_default(&map, &agents, 20).unwrap();

        assert_eq!(result.paths.len(), 2);
        assert_eq!(result.paths[0].last(), Some(&(4, 0)));
        assert_eq!(result.paths[1].last(), Some(&(4, 4)));
        assert!(verify_no_conflicts(&result.paths));
    }

    #[test]
    fn test_priority_planning_crossing_paths() {
        let map = GridMap::new(5, 5);
        let agents = vec![
            Agent {
                start: (0, 2),
                goal: (4, 2),
            },
            Agent {
                start: (2, 0),
                goal: (2, 4),
            },
        ];

        let result = priority_planning_default(&map, &agents, 20).unwrap();

        assert_eq!(result.paths[0].last(), Some(&(4, 2)));
        assert_eq!(result.paths[1].last(), Some(&(2, 4)));
        assert!(verify_no_conflicts(&result.paths));
    }

    #[test]
    fn test_priority_planning_head_on() {
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

        let result = priority_planning_default(&map, &agents, 20).unwrap();

        assert_eq!(result.paths[0].last(), Some(&(4, 1)));
        assert_eq!(result.paths[1].last(), Some(&(0, 1)));
        assert!(verify_no_conflicts(&result.paths));
    }

    #[test]
    fn test_priority_planning_single_agent() {
        let map = GridMap::new(5, 5);
        let agents = vec![Agent {
            start: (0, 0),
            goal: (3, 3),
        }];

        let result = priority_planning_default(&map, &agents, 20).unwrap();

        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].last(), Some(&(3, 3)));
        assert_eq!(result.total_cost, 6); // Manhattan distance
    }

    #[test]
    fn test_priority_planning_empty() {
        let map = GridMap::new(5, 5);
        let agents: Vec<Agent> = vec![];

        let result = priority_planning_default(&map, &agents, 20).unwrap();

        assert!(result.paths.is_empty());
        assert_eq!(result.total_cost, 0);
    }

    #[test]
    fn test_priority_planning_with_obstacles() {
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

        let result = priority_planning_default(&map, &agents, 30).unwrap();

        assert_eq!(result.paths[0].last(), Some(&(4, 2)));
        assert_eq!(result.paths[1].last(), Some(&(1, 4)));
        assert!(verify_no_conflicts(&result.paths));
    }

    #[test]
    fn test_priority_planning_three_agents() {
        let map = GridMap::new(7, 7);
        let agents = vec![
            Agent {
                start: (0, 0),
                goal: (6, 0),
            },
            Agent {
                start: (0, 3),
                goal: (6, 3),
            },
            Agent {
                start: (0, 6),
                goal: (6, 6),
            },
        ];

        let result = priority_planning_default(&map, &agents, 30).unwrap();

        assert_eq!(result.paths.len(), 3);
        assert_eq!(result.paths[0].last(), Some(&(6, 0)));
        assert_eq!(result.paths[1].last(), Some(&(6, 3)));
        assert_eq!(result.paths[2].last(), Some(&(6, 6)));
        assert!(verify_no_conflicts(&result.paths));
    }

    #[test]
    fn test_different_strategies() {
        let map = GridMap::new(5, 5);
        let agents = vec![
            Agent {
                start: (0, 0),
                goal: (1, 0),
            }, // short path
            Agent {
                start: (0, 4),
                goal: (4, 4),
            }, // long path
        ];

        // With LongestPathFirst, agent 1 (long) should plan first
        let result_long =
            priority_planning(&map, &agents, 20, PriorityStrategy::LongestPathFirst).unwrap();
        assert_eq!(result_long.planning_order[0], 1);

        // With ShortestPathFirst, agent 0 (short) should plan first
        let result_short =
            priority_planning(&map, &agents, 20, PriorityStrategy::ShortestPathFirst).unwrap();
        assert_eq!(result_short.planning_order[0], 0);

        // Both should produce valid solutions
        assert!(verify_no_conflicts(&result_long.paths));
        assert!(verify_no_conflicts(&result_short.paths));
    }

    #[test]
    fn test_input_order_strategy() {
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
            Agent {
                start: (2, 2),
                goal: (2, 0),
            },
        ];

        let result = priority_planning(&map, &agents, 20, PriorityStrategy::InputOrder).unwrap();

        assert_eq!(result.planning_order, vec![0, 1, 2]);
        assert!(verify_no_conflicts(&result.paths));
    }
}
