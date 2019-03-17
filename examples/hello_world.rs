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
    let encoding:Vec<_> = "abcdefghijklmnopqrstuvwxyz ".chars().collect();
    let test = "glue";

    loop {
        let mut fitness = HashMap::new();
        for species in &pop {
            for genome in &species.2 {
                let mut all_pass = true;
                let mut phenotype = Phenotype::from_genome(genome);
                for (i, letter) in test.chars().enumerate() {
                    let truth = encoding.iter().position(|c| *c == letter).unwrap() as f32;
                    let truth = truth / (encoding.len() as f32);
                    let output = phenotype.evaluate(&[0.0]);
                    let idx = (output[0] * (encoding.len() as f32)).round() as usize;
                    let idx = idx.max(0).min(encoding.len()-1);
                    if encoding[idx] != letter {
                        all_pass = false;
                    }
                    let f = 1.0 - (output[0] - truth).abs() + if encoding[idx] == letter { 10.0 } else { 0.0 };
                    
                    let e = fitness.entry(genome.genome_id).or_insert(0.0);
                    *e += f;
                }
                if all_pass {
                    eprintln!("Done in {} generations: {:?}", generation, genome);
                    let mut phenotype = Phenotype::from_genome(genome);
                    for i in 0..test.chars().count() {
                        let output = phenotype.evaluate(&[0.0]);
                        let idx = (output[0] * (encoding.len() as f32)).round() as usize;
                        let idx = idx.max(0).min(encoding.len()-1);
                        eprintln!("{}", encoding[idx]);
                    }
                    return;
                }
            }
        }
        let fitness: HashMap<usize, f32> = fitness.iter().map(|(k, v)| (*k, v.powf(2.0))).collect();
        if generation % 100 == 0 {
            let (most_fit, _) = fitness.iter().max_by_key(|(k,v)| (*v * 1000.0) as i32).unwrap();
            for s in &pop {
                for genome in &s.2 {
                    if genome.genome_id == *most_fit {
                        let mut phenotype = Phenotype::from_genome(genome);
                        for i in 0..test.chars().count() {
                            let output = phenotype.evaluate(&[0.0]);
                            let idx = (output[0] * (encoding.len() as f32)).round() as usize;
                            let idx = idx.max(0).min(encoding.len()-1);
                            eprintln!("{}", encoding[idx]);
                        }
                        break
                    }
                }
            }
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
