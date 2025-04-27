use geo::{Coord, Distance, EuclideanDistance, LineString, Rect};
use geo::algorithm::line_measures::Euclidean;
use itertools::Itertools;
use lib_ppa_derive::FromParams;
use rand::prelude::StdRng;
use rand::SeedableRng;
use crate::from_params::{FromParams, ParamType, ParamValue, ParamInfo};
use crate::path_planner::classic::{coords_to_path, sample_random_position};
use crate::path_planner::PathPlanner;
use crate::register_path_planner;
use crate::representation::paths::PathResult;
use crate::representation::world::World;
use kiddo::{KdTree, SquaredEuclidean};

#[derive(PartialEq, Clone, Copy, Debug, FromParams)]
pub struct RRT {
    #[param(
        description = "Step Size",
        default_value = 20.0,
        min_value = 0.1,
    )]
    pub step_size: f32,

    #[param(
        description = "Max Vertices of the Tree",
        default_value = 10000,
        min_value = 1,
    )]
    pub max_vertices: usize,
}
register_path_planner!(RRT);



#[derive(Clone, Copy)]
struct Node {
    position: Coord<f32>,
    parent: Option<usize>,
}


impl RRT {
    pub fn compute_path_with_history(&self, start: Coord<f32>, end: Coord<f32>, world: &World, seed: u64, history_opt: &mut Option<Vec<Vec<PathResult>>>) -> PathResult {
        let mut rng = StdRng::seed_from_u64(seed);
        let world_bounds: Rect<f32> = world.bbox;

        let mut add_history_fn = |tree: &Vec<Node>, new_node_index: usize| {
            if let Some(history) = history_opt.as_mut() {
                if history.is_empty() {
                    history.push(vec![]);
                }
                let parent_index = tree[new_node_index].parent.unwrap();
                let parent_position = tree[parent_index].position;
                let new_position = tree[new_node_index].position;
                let line_string = LineString(vec![parent_position, new_position]);

                // add previous path result and new line to history
                let mut new_history = history.last().unwrap().clone();
                new_history.push(world.get_path_result(&line_string));
                history.push(new_history);
            }
        };

        // initialize tree with start node
        let mut tree: Vec<Node> = vec![Node { position: start, parent: None }];
        // initialize kdTree with start node for nearest neighbor search
        let mut kdtree: KdTree<f32, 2> = KdTree::new();
        kdtree.add(&[start.x, start.y], 0); // Start node at index 0

        // loop until end node is reached or max iterations are reached
        for _ in 0..self.max_vertices {
            // generate random point
            let random_point = sample_random_position(&mut rng, world_bounds);

            // find nearest node in tree
            let nearest_index = kdtree.nearest_one::<SquaredEuclidean>(&[random_point.x, random_point.y]).item as usize;

            // extend tree towards random point
            let new_position = extend(&tree[nearest_index ], random_point, self.step_size);

            if is_collision_free(world, tree[nearest_index].position, new_position) {
                let new_index = tree.len();
                tree.push(Node { position: new_position, parent: Some(nearest_index) });
                kdtree.add(&[new_position.x, new_position.y], new_index as u64);

                // Add history for the new node
                add_history_fn(&tree, new_index);

                // check if end node is reached
                if check_goal_reached(&tree[new_index], end, self.step_size, world) {
                    // add goal to tree
                    let goal_index = tree.len();
                    tree.push(Node { position: end, parent: Some(new_index) });

                    // Add history for the goal node
                    add_history_fn(&tree, goal_index);

                    // return path
                    return reconstruct_path(&tree, goal_index, world);
                }
            }
        }

        print!("No path found, returning empty path");
        // return empty path if no path is found //todo: return something else that's more meaningful
        let line_string = LineString(
            std::iter::once(start)
                .collect()
        );
        world.get_path_result(&line_string)
    }
}

fn check_goal_reached(node: &Node, goal: Coord<f32>, step_size: f32, world: &World) -> bool {
    // check if goal is reachable with direct connection
    geo::algorithm::line_measures::Euclidean::distance(node.position, goal) < step_size && is_collision_free(world, node.position, goal)
}

fn extend(node: &Node, target: Coord<f32>, step_size: f32) -> Coord<f32> {
    let direction = (
        target.x - node.position.x,
        target.y - node.position.y,
    );
    let length = (direction.0.powi(2) + direction.1.powi(2)).sqrt();

    if length < step_size {
        return target; // Directly connect if within step size
    }

    // Extend towards target
    Coord {
        x: node.position.x + direction.0 / length * step_size,
        y: node.position.y + direction.1 / length * step_size,
    }
}

fn is_collision_free(world: &World, from: Coord<f32>, to: Coord<f32>) -> bool {
    let line = geo::Line::new(
        geo::Point::new(from.x, from.y),
        geo::Point::new(to.x, to.y),
    );
    world.get_intersecting_obstacles(line).is_empty()
}

fn reconstruct_path(tree: &Vec<Node>, goal_index: usize, world: &World) -> PathResult {
    let mut path = vec![];
    let mut current = Some(goal_index);

    while let Some(index) = current {
        path.push(tree[index].position);
        current = tree[index].parent;
    }

    path.reverse(); // Start to goal order
    let line_string = LineString(path);
    world.get_path_result(&line_string)
}



impl PathPlanner for RRT {
    fn compute_path(&self, start: Coord<f32>, end: Coord<f32>, world: &World, seed: u64) -> PathResult {
        Self::compute_path_with_history(self, start, end, world, seed, &mut None)
    }
    fn compute_path_with_history(&self, start: Coord<f32>, end: Coord<f32>, world: &World, seed: u64, history_opt: &mut Option<Vec<Vec<PathResult>>>) -> PathResult {
        Self::compute_path_with_history(self, start, end, world, seed, history_opt)
    }
}