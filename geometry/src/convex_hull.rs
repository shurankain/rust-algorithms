#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    fn orientation(a: Point, b: Point, c: Point) -> i32 {
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    }

    fn distance_squared(a: Point, b: Point) -> i32 {
        (a.x - b.x).pow(2) + (a.y - b.y).pow(2)
    }
}

pub fn convex_hull(mut points: Vec<Point>) -> Vec<Point> {
    if points.len() < 3 {
        return points;
    }

    // 1. Looking for a point with minimal y (and x if equals)
    let start = *points.iter().min_by_key(|p| (p.y, p.x)).expect("No points");

    // 2. Sorting by an angle and distance
    points.sort_by(|a, b| {
        let orient = Point::orientation(start, *a, *b);
        if orient == 0 {
            Point::distance_squared(start, *a).cmp(&Point::distance_squared(start, *b))
        } else {
            orient.cmp(&0).reverse()
        }
    });

    // 3. Stack: checking the orientation for each triplet
    let mut hull = vec![start];
    for &point in points.iter().skip(1) {
        while hull.len() >= 2
            && Point::orientation(hull[hull.len() - 2], hull[hull.len() - 1], point) <= 0
        {
            hull.pop();
        }
        hull.push(point);
    }

    hull
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convex_hull_square() {
        let points = vec![
            Point { x: 0, y: 0 },
            Point { x: 1, y: 1 },
            Point { x: 0, y: 1 },
            Point { x: 1, y: 0 },
        ];

        let hull = convex_hull(points);

        // Expecting a square: 0,0 → 1,0 → 1,1 → 0,1
        assert_eq!(
            hull,
            vec![
                Point { x: 0, y: 0 },
                Point { x: 1, y: 0 },
                Point { x: 1, y: 1 },
                Point { x: 0, y: 1 },
            ]
        );
    }
}
