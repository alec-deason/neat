extern crate neat;
extern crate rand;

use rand::Rng;

use rand::prelude::SliceRandom;

use std::collections::HashMap;

use neat::{initial_population, next_generation, Genome, Phenotype};

fn main() {
    let mut rng = rand::thread_rng();
    let mut pop = initial_population(150, 2, 1);
    let mut generation = 0;

    loop {
        let mut best_fail = HashMap::new();
        let mut fitness = HashMap::new();
        for species in &pop {
            for genome in &species.2 {
                let mut all_pass = true;
                for (a, b) in &[(true, true), (true, false), (false, true), (false, false)] {
                    let a = *a;
                    let b = *b;
                    let mut phenotype = Phenotype::from_genome(genome);
                    let truth = if (a || b) && !(a && b) { 1.0 } else { 0.0 };
                    //let truth = if !(a&&b) { 1.0 } else { 0.0 };
                    let output =
                        phenotype.evaluate(&[if a { 1.0 } else { 0.0 }, if b { 1.0 } else { 0.0 }]);
                    let f = 1.0 - (output[0] - truth).abs();
                    //let ok = (output[0] > 0.0 && truth == 1.0) ||
                    //         (output[0] <= 0.0 && truth == 0.0);
                    let ok = output[0].round() == truth;
                    if !ok {
                        all_pass = false;
                        best_fail.entry(genome.genome_id).or_insert(vec![]).push((a, b, (output[0] - truth, output[0])));
                    }
                    let e = fitness.entry(genome.genome_id).or_insert(0.0);
                    *e += f;
                }
                if all_pass {
                    eprintln!("Done in {} generations: {:?}", generation, genome);
                    return;
                }
            }
        }
        let fitness: HashMap<usize, f32> = fitness.iter().map(|(k, v)| (*k, v.powf(2.0))).collect();
        let (most_fit, _) = fitness.iter().max_by_key(|(k,v)| (*v * 1000.0) as i32).unwrap();
        /*
        for s in &pop {
            for genome in &s.2 {
                if genome.genome_id == *most_fit {
                    let mut phenotype = Phenotype::from_genome(genome);
                    phenotype.to_dot();
                    break
                }
            }
        }
        */
        let mut sorted_fitness:Vec<f32> = fitness
            .values()
            .map(|v| v.sqrt())
            .collect();
        sorted_fitness.sort_by_key(|v| (v * 10000.0) as i32);
        let mean_fitness = sorted_fitness.iter().sum::<f32>() / sorted_fitness.len() as f32;
        let median_fitness = sorted_fitness[(sorted_fitness.len() as f32 / 2.0) as usize];
        let max_fitness = sorted_fitness[sorted_fitness.len() -1];
        let min_fitness = sorted_fitness[0];
        pop = next_generation(&fitness, &pop);
        eprintln!(
            "Generation: {} Species: {} Min Fitness: {} Median Fitness: {} Mean Fitness: {} Max Fitness: {}",
            generation,
            pop.len(),
            min_fitness,
            median_fitness,
            mean_fitness,
            max_fitness,
        );
        generation += 1;
        /*
        let mut rng = rand::thread_rng();
        let choice = pop.choose(&mut rng).unwrap();
        let choice = choice.choose(&mut rng).unwrap();
        eprintln!("{:?}", choice);
        */
    }
}
