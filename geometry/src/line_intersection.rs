/// Determine the orientation of the ordered triplet (p, q, r).
/// Returns:
/// - 0 if collinear
/// - 1 if clockwise
/// - 2 if counterclockwise
fn orientation(p: (i32, i32), q: (i32, i32), r: (i32, i32)) -> i32 {
    let val = (q.1 - p.1) * (r.0 - q.0) - (q.0 - p.0) * (r.1 - q.1);
    if val == 0 {
        0 // collinear
    } else if val > 0 {
        1 // clockwise
    } else {
        2 // counterclockwise
    }
}

/// Check if point `p` lies on the segment `seg_start`-`seg_end`
fn on_segment(p: (i32, i32), seg_start: (i32, i32), seg_end: (i32, i32)) -> bool {
    p.0 >= seg_start.0.min(seg_end.0)
        && p.0 <= seg_start.0.max(seg_end.0)
        && p.1 >= seg_start.1.min(seg_end.1)
        && p.1 <= seg_start.1.max(seg_end.1)
}

/// Returns true if segments (p1,q1) and (p2,q2) intersect
pub fn segments_intersect(p1: (i32, i32), q1: (i32, i32), p2: (i32, i32), q2: (i32, i32)) -> bool {
    let o1 = orientation(p1, q1, p2);
    let o2 = orientation(p1, q1, q2);
    let o3 = orientation(p2, q2, p1);
    let o4 = orientation(p2, q2, q1);

    // General case
    if o1 != o2 && o3 != o4 {
        return true;
    }

    // Special cases
    if o1 == 0 && on_segment(p2, p1, q1) {
        return true;
    }
    if o2 == 0 && on_segment(q2, p1, q1) {
        return true;
    }
    if o3 == 0 && on_segment(p1, p2, q2) {
        return true;
    }
    if o4 == 0 && on_segment(q1, p2, q2) {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersecting_segments() {
        assert!(segments_intersect((1, 1), (4, 4), (1, 4), (4, 1)));
    }

    #[test]
    fn test_non_intersecting_segments() {
        assert!(!segments_intersect((1, 1), (2, 2), (3, 3), (4, 4)));
    }

    #[test]
    fn test_collinear_overlap() {
        assert!(segments_intersect((1, 1), (4, 1), (2, 1), (3, 1)));
    }

    #[test]
    fn test_endpoint_touch() {
        assert!(segments_intersect((0, 0), (2, 2), (2, 2), (4, 0)));
    }
}
