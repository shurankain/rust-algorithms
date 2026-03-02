pub mod convex_hull;
pub mod line_intersection;
pub mod sweep_line;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Cross product of vectors (a->b) and (a->c)
    /// Returns positive if counterclockwise, negative if clockwise, 0 if collinear
    pub fn cross(a: Point, b: Point, c: Point) -> i32 {
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    }

    pub fn distance_squared(a: Point, b: Point) -> i32 {
        (a.x - b.x).pow(2) + (a.y - b.y).pow(2)
    }
}
