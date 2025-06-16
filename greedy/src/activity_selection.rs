// Activity Selection Problem using greedy algorithm
#[derive(Debug, Clone)]
pub struct Activity {
    pub start: usize,
    pub end: usize,
}

pub fn activity_selection(mut activities: Vec<Activity>) -> Vec<Activity> {
    // Sort activities by end time
    activities.sort_by_key(|a| a.end);

    let mut selected = Vec::new();
    let mut last_end = 0;

    for activity in activities {
        if activity.start >= last_end {
            last_end = activity.end;
            selected.push(activity);
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activity_selection() {
        let activities = vec![
            Activity { start: 1, end: 2 },
            Activity { start: 3, end: 4 },
            Activity { start: 0, end: 6 },
            Activity { start: 5, end: 7 },
            Activity { start: 8, end: 9 },
            Activity { start: 5, end: 9 },
        ];

        let result = activity_selection(activities);
        let expected = [
            Activity { start: 1, end: 2 },
            Activity { start: 3, end: 4 },
            Activity { start: 5, end: 7 },
            Activity { start: 8, end: 9 },
        ];

        assert_eq!(result.len(), expected.len());
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_eq!(a.start, b.start);
            assert_eq!(a.end, b.end);
        }
    }
}
