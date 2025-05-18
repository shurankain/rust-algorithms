use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub id: usize,
    pub p1: Point,
    pub p2: Point,
}

// Simplified function, cheking the intersection of two segments till the borders precision
fn segments_intersect(a: Segment, b: Segment) -> bool {
    fn orientation(p: Point, q: Point, r: Point) -> i32 {
        let val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
        if val.abs() < 1e-9 { 0 } else if val > 0.0 { 1 } else { 2 }
    }

    fn on_segment(p: Point, q: Point, r: Point) -> bool {
        q.x.min(r.x) <= p.x && p.x <= q.x.max(r.x) &&
        q.y.min(r.y) <= p.y && p.y <= q.y.max(r.y)
    }

    let (p1, q1, p2, q2) = (a.p1, a.p2, b.p1, b.p2);

    let o1 = orientation(p1, q1, p2);
    let o2 = orientation(p1, q1, q2);
    let o3 = orientation(p2, q2, p1);
    let o4 = orientation(p2, q2, q1);

    if o1 != o2 && o3 != o4 {
        return true;
    }

    o1 == 0 && on_segment(p2, p1, q1)
        || o2 == 0 && on_segment(q2, p1, q1)
        || o3 == 0 && on_segment(p1, p2, q2)
        || o4 == 0 && on_segment(q1, p2, q2)
}

// Main function, retuns pairs of indexes of intersecting segments
pub fn find_intersections(segments: &[Segment]) -> Vec<(usize, usize)> {
    let mut events = Vec::new();
    for seg in segments {
        let (left, right) = if seg.p1.x < seg.p2.x {
            (seg.p1, seg.p2)
        } else {
            (seg.p2, seg.p1)
        };
        events.push((left.x, true, *seg));  // start event
        events.push((right.x, false, *seg)); // end event
    }

    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut active: BTreeSet<usize> = BTreeSet::new();
    let mut intersections = Vec::new();

    for (_, is_start, seg) in events {
        if is_start {
            for &other_id in &active {
                if segments_intersect(seg, segments[other_id]) {
                    intersections.push((seg.id, other_id));
                }
            }
            active.insert(seg.id);
        } else {
            active.remove(&seg.id);
        }
    }

    intersections
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersections() {
        let segments = vec![
            Segment { id: 0, p1: Point { x: 1.0, y: 1.0 }, p2: Point { x: 4.0, y: 4.0 } },
            Segment { id: 1, p1: Point { x: 1.0, y: 4.0 }, p2: Point { x: 4.0, y: 1.0 } },
            Segment { id: 2, p1: Point { x: 5.0, y: 5.0 }, p2: Point { x: 6.0, y: 6.0 } },
        ];

        let result = find_intersections(&segments);
        assert_eq!(result, vec![(1, 0)]);
    }
}