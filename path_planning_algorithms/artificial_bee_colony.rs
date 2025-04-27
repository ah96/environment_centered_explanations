use crate::from_params::FromParams;
use crate::from_params::ParamInfo;
use crate::from_params::ParamType;
use crate::from_params::ParamValue;
use crate::path_planner::meta_heuristic::{clamp, coords_to_path, gen_random_path};
use crate::path_planner::PathPlanner;
use crate::register_path_planner;
use crate::representation::paths::PathResult;
use crate::representation::world::World;
use geo::{Coord, Rect};
use itertools::Itertools;
use lib_ppa_derive::FromParams;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromParams)]
pub struct ArtificialBeeColony {
    #[param(
        description = "Amount of employed bees. Should be equal n_onlooker_bees.",
        default_value = 100,
        min_value = 1,
    )]
    pub n_employed_bees: usize,

    #[param(
        description = "Amount of onlooker bees. Should be equal n_employed_bees.",
        default_value = 100,
        min_value = 1,
    )]
    pub n_onlooker_bees: usize,

    #[param(
        description = "Maximum amount of failed tries before a honey source is abandoned.",
        default_value = 20,
        min_value = 1,
    )]
    pub limit: usize,

    #[param(
        description = "Amount of iterations.",
        default_value = 100,
        min_value = 1,
    )]
    pub n_itr: usize,

    #[param(
        description = "Amount of intermediate way points on the path.",
        default_value = 20,
        min_value = 1,
    )]
    pub n_points: usize,
}
register_path_planner!(ArtificialBeeColony);


#[derive(Clone, Debug, PartialEq)]
struct HoneySource {
    pub solution: Vec<Coord<f32>>,
    pub fitness: f32,
    pub n_tries: usize,
}

impl Eq for HoneySource {}

impl PartialOrd for HoneySource {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.fitness.partial_cmp(&other.fitness)
    }
}

impl Ord for HoneySource {
    fn cmp(&self, other: &Self) -> Ordering {
        self.fitness.partial_cmp(&other.fitness).unwrap()
    }
}

impl HoneySource {
    fn gen_new_solution(&self, other_solution: &Vec<Coord<f32>>, bbox: &Rect<f32>, rng: &mut StdRng) -> Vec<Coord<f32>> {
        let phi: f32 = rng.gen_range(-1.0..=1.0);
        let dim_idx = rng.gen_range(0..self.solution.len() * 2);
        let coord_idx = dim_idx / 2;
        
        let mut new_solution = self.solution.clone();
        if dim_idx % 2 == 0 {
            new_solution[coord_idx].x += phi * (new_solution[coord_idx].x - other_solution[coord_idx].x);
        } else {
            new_solution[coord_idx].y += phi * (new_solution[coord_idx].y - other_solution[coord_idx].y);
        }
        clamp(&mut new_solution[coord_idx], bbox);
        
        new_solution
    }
    
    fn update(&mut self, new_solution: Vec<Coord<f32>>, start: Coord<f32>, end: Coord<f32>, world: &World) {
        let path = coords_to_path(&new_solution, start, end);
        let new_fitness = world.get_path_result(&path).get_fitness();
        
        if new_fitness < self.fitness {
            self.solution = new_solution;
            self.fitness = new_fitness;
            self.n_tries = 0;
        } else { 
            self.n_tries += 1;
        }
    }
}

impl ArtificialBeeColony {
    pub fn compute_path_with_history(&self, start: Coord<f32>, end: Coord<f32>, world: &World, seed: u64, history_opt: &mut Option<Vec<Vec<PathResult>>>) -> PathResult {
        let mut rng = StdRng::seed_from_u64(seed);
        let get_fitness_fn = |solution: &Vec<Coord<f32>>| {
            let path = coords_to_path(solution, start, end);
            let fitness = world.get_path_result(&path).get_fitness();
            fitness
        };
        let mut gen_random_honey_source = || {
            let solution = gen_random_path(&mut rng, world.bbox, self.n_points);
            let fitness = get_fitness_fn(&solution);
            HoneySource { solution, fitness, n_tries: 0 }
        };
        let mut add_history_fn = |honey_sources: &Vec<HoneySource>| {
            history_opt.as_mut().map(|history| {
                history.push(
                    honey_sources.iter()
                        .map(|hs| coords_to_path(&hs.solution, start, end))
                        .map(|path| world.get_path_result(&path))
                        .collect_vec()
                );
            });
        };

        let mut honey_sources = (0..self.n_employed_bees)
            .map(|_| gen_random_honey_source())
            .collect_vec();
        add_history_fn(&honey_sources);
        let mut best_honey_source = honey_sources.iter().min().unwrap().clone();

        for _curr_itr in 2..=self.n_itr {
            // 1. Employed Bee Phase: Update each Honey Source exactly once
            for honey_source_idx in 0..honey_sources.len() {
                let other_honey_source = Self::get_rand_elem_for_idx(&honey_sources, honey_source_idx, &mut rng);
                let new_solution = honey_sources[honey_source_idx].gen_new_solution(&other_honey_source.solution, &world.bbox, &mut rng);
                honey_sources[honey_source_idx].update(new_solution, start, end, world);
            }

            // 2. Onlooker Bee Phase: Update better Honey Sources more often 
            let honey_source_idx_selection = Self::roulette_idx_selection(
                &honey_sources.iter().map(|hs| hs.fitness).collect_vec(),
                self.n_onlooker_bees,
                &mut rng,
            );
            honey_source_idx_selection.into_iter().for_each(|honey_source_idx| {
                let other_honey_source = Self::get_rand_elem_for_idx(&honey_sources, honey_source_idx, &mut rng);
                let new_solution = honey_sources[honey_source_idx].gen_new_solution(&other_honey_source.solution, &world.bbox, &mut rng);
                honey_sources[honey_source_idx].update(new_solution, start, end, world);
            });

            // 3. Save best Solution
            let curr_best_honey_source = honey_sources.iter().min().unwrap();
            if curr_best_honey_source < &best_honey_source {
                best_honey_source = curr_best_honey_source.clone();
            }

            // 4. Scout Bee Phase: Generate new random Honey Sources that were depleted
            honey_sources.iter_mut()
                .filter(|hs| hs.n_tries >= self.limit)
                .for_each(|honey_source| {
                    let solution = gen_random_path(&mut rng, world.bbox, self.n_points);
                    let fitness = get_fitness_fn(&solution);

                    honey_source.solution = solution;
                    honey_source.fitness = fitness;
                    honey_source.n_tries = 0;
                });

            add_history_fn(&honey_sources);
        }

        world.get_path_result(&coords_to_path(&best_honey_source.solution, start, end))
    }
    
    fn get_rand_elem_for_idx<'a, T>(vec: &'a Vec<T>, idx: usize, rng: &mut StdRng) -> &'a T {
        let rand_idx = rng.gen_range(0..vec.len()-1);
        if rand_idx != idx {
            &vec[rand_idx]
        } else { 
            &vec[vec.len() - 1]
        }
    }
    
    fn roulette_idx_selection(pop_fitness: &Vec<f32>, num: usize, rng: &mut StdRng) -> Vec<usize> {
        // As bigger fitness is worse but in roulette selection bigger fitness is more likely to be selected, fitness values must be normalized 
        let pop_fitness_normalized = pop_fitness.iter().map(|f| 1.0 / f).collect_vec();
        
        let fitness_total = pop_fitness_normalized.iter().sum::<f32>();
        let fitness_props = pop_fitness_normalized.iter().map(|f| f / fitness_total).collect_vec();

        (0..num).map(|_| {
            // Generate random number between 0 and 1, marking the spot on the roulette wheel
            let r = rng.gen::<f32>();
            let mut acc = 0.0;
            for (idx, fitness_prop) in fitness_props.iter().enumerate() {
                acc += fitness_prop;
                if r <= acc {
                    return idx;
                }
            }
            pop_fitness.len() - 1
        }).collect_vec()
    }
}


impl PathPlanner for ArtificialBeeColony {
    fn compute_path(&self, start: Coord<f32>, end: Coord<f32>, world: &World, seed: u64) -> PathResult {
        Self::compute_path_with_history(self, start, end, world, seed, &mut None)
    }
    fn compute_path_with_history(&self, start: Coord<f32>, end: Coord<f32>, world: &World, seed: u64, history_opt: &mut Option<Vec<Vec<PathResult>>>) -> PathResult {
        Self::compute_path_with_history(self, start, end, world, seed, history_opt)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::representation::world::obstacle::circle::Circle;
    use crate::representation::world::obstacle::Obstacle;
    use geo::{coord, point, Rect};

    #[test]
    fn test_artificial_bee_colony() {
        let world = World::new(
            Rect::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 100.0, y: 100.0 }),
            vec![
                Obstacle::Circle(Circle { center: point! { x: 50.0, y: 50.0}, radius: 20. })
            ]
        );

        let abc = ArtificialBeeColony {
            n_employed_bees: 300,
            n_onlooker_bees: 300,
            limit: 30,
            n_itr: 1000,
            n_points: 2,
        };
        let start = world.bbox.min();
        let end = world.bbox.max();
        
        abc.compute_path(start, end, &world, 42);
    }
}
