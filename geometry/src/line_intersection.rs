use crate::Point;

/// Orientation using cross product sign:
/// - 0 if collinear
/// - 1 if clockwise
/// - 2 if counterclockwise
fn orientation(p: Point, q: Point, r: Point) -> i32 {
    let val = Point::cross(p, q, r);
    if val == 0 {
        0
    } else if val < 0 {
        1 // clockwise
    } else {
        2 // counterclockwise
    }
}

/// Check if point `p` lies on the segment `seg_start`-`seg_end`
fn on_segment(p: Point, seg_start: Point, seg_end: Point) -> bool {
    p.x >= seg_start.x.min(seg_end.x)
        && p.x <= seg_start.x.max(seg_end.x)
        && p.y >= seg_start.y.min(seg_end.y)
        && p.y <= seg_start.y.max(seg_end.y)
}

/// Returns true if segments (p1,q1) and (p2,q2) intersect
pub fn segments_intersect(p1: Point, q1: Point, p2: Point, q2: Point) -> bool {
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

    fn p(x: i32, y: i32) -> Point {
        Point::new(x, y)
    }

    #[test]
    fn test_intersecting_segments() {
        assert!(segments_intersect(p(1, 1), p(4, 4), p(1, 4), p(4, 1)));
    }

    #[test]
    fn test_non_intersecting_segments() {
        assert!(!segments_intersect(p(1, 1), p(2, 2), p(3, 3), p(4, 4)));
    }

    #[test]
    fn test_collinear_overlap() {
        assert!(segments_intersect(p(1, 1), p(4, 1), p(2, 1), p(3, 1)));
    }

    #[test]
    fn test_endpoint_touch() {
        assert!(segments_intersect(p(0, 0), p(2, 2), p(2, 2), p(4, 0)));
    }
}
