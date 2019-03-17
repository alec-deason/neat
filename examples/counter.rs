extern crate neat;
extern crate rand;

use rand::Rng;

use rand::prelude::SliceRandom;

use std::collections::HashMap;

use neat::{initial_population, next_generation, Genome, Phenotype};

fn main() {
    let mut rng = rand::thread_rng();
    let mut pop = initial_population(150, 1, 1);
    let mut generation = 0;
    let len = 20;

    loop {
        let mut fitness = HashMap::new();
        for species in &pop {
            for genome in &species.2 {
                let mut all_pass = true;
                for _ in 0..10 {
                    let mut phenotype = Phenotype::from_genome(genome);
                    let mut true_count = 0;
                    for _ in 0..len {
                        let input = if rng.gen::<bool>() { 1 } else { 0 };
                        let output = phenotype.evaluate(&[input as f32]);
                        true_count += input;
                        if (output[0] * len as f32).round() as usize != true_count {
                            all_pass = false;
                        }
                        let f = 1.0 - (true_count as f32 / len as f32 - output[0]).abs();
                        
                        let e = fitness.entry(genome.genome_id).or_insert(0.0);
                        *e += f;
                    }
                }
                if all_pass {
                    eprintln!("Done in {} generations: {:?}", generation, genome);
                    return;
                }
            }
        }
        let fitness: HashMap<usize, f32> = fitness.iter().map(|(k, v)| (*k, v.powf(2.0))).collect();
        if generation % 100 == 0 {
            let mut sorted_fitness:Vec<f32> = fitness
                .values()
                .map(|v| v.sqrt())
                .collect();
            sorted_fitness.sort_by_key(|v| (v * 10000.0) as i32);
            let mean_fitness = sorted_fitness.iter().sum::<f32>() / sorted_fitness.len() as f32;
            let median_fitness = sorted_fitness[(sorted_fitness.len() as f32 / 2.0) as usize];
            let max_fitness = sorted_fitness[sorted_fitness.len() -1];
            let min_fitness = sorted_fitness[0];
            eprintln!(
                "Generation: {} Species: {} Min Fitness: {} Median Fitness: {} Mean Fitness: {} Max Fitness: {}",
                generation,
                pop.len(),
                min_fitness,
                median_fitness,
                mean_fitness,
                max_fitness,
            );
        }
        pop = next_generation(&fitness, &pop);
        generation += 1;
    }
}
