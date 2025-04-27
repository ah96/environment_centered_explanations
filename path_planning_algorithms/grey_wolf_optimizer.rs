use crate::from_params::FromParams;
use crate::from_params::ParamInfo;
use crate::from_params::ParamType;
use crate::from_params::ParamValue;
use crate::path_planner::meta_heuristic::{clamp, coords_to_path, gen_random_path, get_pop_fitness};
use crate::path_planner::PathPlanner;
use crate::register_path_planner;
use crate::representation::paths::PathResult;
use crate::representation::world::World;
use crate::utils::n_smallest_iterator::NSmallestByIterator;
use geo::{Coord, Rect};
use itertools::Itertools;
use lib_ppa_derive::FromParams;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::ops::{Add, Div};

const C_CONSTANT: f32 = 2.0;

#[derive(PartialEq, Clone, Copy, Debug, FromParams)]
pub struct GreyWolfOptimizer {
    #[param(
        description = "Population Size",
        default_value = 50,
        min_value = 3,
    )]
    pub pop_size: usize,

    #[param(
        description = "Amount of Iterations",
        default_value = 100,
        min_value = 1,
    )]
    pub n_itr: usize,

    #[param(
        description = "Amount of intermediate Way Points on the Path",
        default_value = 20,
        min_value = 1,
    )]
    pub n_points: usize,
}
register_path_planner!(GreyWolfOptimizer);


impl GreyWolfOptimizer {
    pub fn compute_path(&self, start: Coord<f32>, end: Coord<f32>, world: &World, seed: u64, history_opt: &mut Option<Vec<Vec<PathResult>>>) -> PathResult {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut a = 2.0;
        
        let mut add_history_fn = |pop: &Vec<Vec<Coord<f32>>>| {
            history_opt.as_mut().map(|history| {
                history.push(
                    pop.iter()
                        .map(|coords| coords_to_path(coords, start, end))
                        .map(|path| world.get_path_result(&path))
                        .collect_vec()
                );
            });
        };

        let mut pop = std::iter::repeat_with(|| gen_random_path(&mut rng, world.bbox, self.n_points))
            .take(self.pop_size)
            .collect_vec();
        add_history_fn(&pop);

        let mut pop_fitness = get_pop_fitness(&pop, start, end, world);
        let mut top_three = pop_fitness.iter().zip(pop.iter())
            .n_smallest_by(3, |(fitness, _)| *fitness)
            .map(|(fitness, path)| (*fitness, path.clone()))
            .collect_vec();
        let mut prey = Self::get_prey(&top_three);

        for curr_itr in 2..=self.n_itr {
            Self::update_a(&mut a, curr_itr, self.n_itr);
            pop = pop.into_iter().map(|wolf| Self::get_next_wolf(&wolf, &prey, a, &world.bbox, &mut rng)).collect_vec();
            add_history_fn(&pop);
            
            pop_fitness = get_pop_fitness(&pop, start, end, world);
            top_three = pop_fitness.iter().zip(pop.iter())
                .chain(top_three.iter().map(|(fitness, path)| (fitness, path)))
                .n_smallest_by(3, |(fitness, _)| *fitness)
                .map(|(fitness, path)| (*fitness, path.clone()))
                .collect_vec();
            prey = Self::get_prey(&top_three);
        }

        let (_best_fitness, best_path) = top_three.first().unwrap();
        world.get_path_result(&coords_to_path(best_path, start, end))
    }
    
    
    fn get_next_wolf(wolf: &Vec<Coord<f32>>, prey: &Vec<Coord<f32>>, a: f32, bbox: &Rect<f32>, rng: &mut StdRng) -> Vec<Coord<f32>> {
        wolf.iter().zip(prey.iter())
            .map(|(wolf_coord, prey_coord)|
                Coord {
                    x: Self::get_next_pos(wolf_coord.x, prey_coord.x, a, rng),
                    y: Self::get_next_pos(wolf_coord.y, prey_coord.y, a, rng),
                }
            )
            .map(|mut coord| *clamp(&mut coord, &bbox))
            .collect_vec()
    }


    #[allow(non_snake_case)]
    #[inline]
    fn get_next_pos(wolf_pos: f32, prey_pos: f32, a: f32, rng: &mut StdRng) -> f32 {
        let A = rng.gen_range(-a..=a);
        let C = C_CONSTANT * rng.gen::<f32>();
        let D = C * prey_pos - wolf_pos;
        prey_pos - A * D
    }


    fn get_prey(top_three: &Vec<(f32, Vec<Coord<f32>>)>) -> Vec<Coord<f32>> {
        let (_, alpha) = &top_three[0];
        let (_, beta) = &top_three[1];
        let (_, delta) = &top_three[2];

        alpha.iter().zip(beta.iter()).zip(delta.iter())
            .map(|((alpha_coord, beta_coord), delta_coord)| [*alpha_coord, *beta_coord, *delta_coord])
            .map(|coords| coords.into_iter().reduce(|c1, c2| c1.add(c2)).unwrap().div(coords.len() as f32))
            .collect_vec()
    }
    

    #[inline]
    fn update_a(a: &mut f32, curr_itr: usize, n_itr: usize) {
        *a = 2.0 * (1.0 - curr_itr as f32 / n_itr as f32)
    }
}


impl PathPlanner for GreyWolfOptimizer {
    fn compute_path(&self, start: Coord<f32>, end: Coord<f32>, world: &World, seed: u64) -> PathResult {
        Self::compute_path(self, start, end, world, seed, &mut None)
    }
    fn compute_path_with_history(&self, start: Coord<f32>, end: Coord<f32>, world: &World, seed: u64, history_opt: &mut Option<Vec<Vec<PathResult>>>) -> PathResult {
        Self::compute_path(self, start, end, world, seed, history_opt)
    }
}