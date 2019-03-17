#![feature(cell_update)]
extern crate rand;
extern crate petgraph;

use petgraph::Graph;
use petgraph::dot::{Dot, Config};
use std::fs::File;
use std::io::prelude::*;

use std::path::Path;
use rand::prelude::IteratorRandom;
use rand::prelude::SliceRandom;
use rand::Rng;
use std::cell::Cell;
use std::collections::HashMap;
use std::f32;
use std::fmt;
use std::rc::Rc;

#[derive(Copy, Clone, Debug)]
enum NodeType {
    Input,
    Hidden,
    Bias,
    Output,
}

#[derive(Copy, Clone, Debug)]
struct NodeGene {
    node_id: usize,
    node_type: NodeType,
}

#[derive(Copy, Clone, Debug)]
struct ConnectionGene {
    innovation_number: usize,
    in_node: usize,
    out_node: usize,
    weight: f32,
    enabled: bool,
}

#[derive(Clone)]
pub struct Genome {
    genome_counter: Rc<Cell<usize>>,
    node_counter: Rc<Cell<usize>>,
    innovation_counter: Rc<Cell<usize>>,
    pub genome_id: usize,
    nodes: HashMap<usize, NodeGene>,
    connections: Vec<ConnectionGene>,
}

impl fmt::Debug for Genome {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Genome {{ genome_id: {}, nodes: {:?}, connections: {:?} }}",
            self.genome_id, self.nodes, self.connections
        )
    }
}

pub struct Phenotype {
    nodes: Vec<f32>,
    bias_node: usize,
    input_nodes: Vec<usize>,
    output_nodes: Vec<usize>,
    connections: Vec<(usize, usize, f32)>,
}

impl Phenotype {
    pub fn from_genome(genome: &Genome) -> Phenotype {
        let mut nodes = vec![];
        let mut input_nodes = vec![];
        let mut output_nodes = vec![];
        let mut bias_node = 0;
        let mut id_to_idx = HashMap::new();

        for node in genome.nodes.values() {
            match node.node_type {
                NodeType::Input => input_nodes.push(nodes.len()),
                NodeType::Bias => bias_node = nodes.len(),
                NodeType::Output => output_nodes.push(nodes.len()),
                _ => (),
            }
            id_to_idx.insert(node.node_id, nodes.len());
            nodes.push(0.0);
        }

        let mut connections = vec![];
        for connection in &genome.connections {
            if connection.enabled {
                connections.push((
                    id_to_idx[&connection.in_node],
                    id_to_idx[&connection.out_node],
                    connection.weight,
                ));
            }
        }

        connections.sort_by_key(|(i, o, _)| {
            if output_nodes.contains(o) {
                100
            } else if input_nodes.contains(i) {
                0
            } else {
                50
            }
        });

        Phenotype {
            nodes,
            bias_node,
            input_nodes,
            output_nodes,
            connections,
        }
    }

    pub fn evaluate(&mut self, inputs: &[f32]) -> Vec<f32> {
        let nodes = &mut self.nodes;
        let connections = &self.connections;
        let input_nodes = &self.input_nodes;
        let output_nodes = &self.output_nodes;
        let bias_node = self.bias_node;

        for (i, v) in inputs.iter().enumerate() {
            nodes[input_nodes[i]] = *v;
        }
        nodes[bias_node] = 1.0;

        //let mut new_nodes = vec![0.0f32; self.nodes.len()];
        for idx in 0..nodes.len() {
            let inputs = connections.iter().filter(|(_, out_node, _)| *out_node == idx).map(|(in_node, _, w)| nodes[*in_node] * w).sum::<f32>();
            let value = 1.0/(1.0+f32::consts::E.powf(-4.9*inputs));
            nodes[idx] = value;
        }
        output_nodes.iter().map(|i| nodes[*i]).collect()
    }

    pub fn to_dot(&self) {
        let mut deps = Graph::<_, f32>::new();
        let mut node_idx = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            let label = if self.input_nodes.contains(&i) {
                "i"
            } else if self.bias_node == i {
                "b"
            } else if self.output_nodes.contains(&i) {
                "o"
            } else {
                "h"
            };
            node_idx.push(deps.add_node(label));
        }
        for (in_node, out_node, weight) in &self.connections {
            deps.extend_with_edges(&[(node_idx[*in_node], node_idx[*out_node], *weight),])
        }

        let mut file = File::create("/tmp/test.dot").unwrap();
        write!(&mut file, "{:?}", Dot::with_config(&deps, &[]));
    }
}

fn add_connection(genome: &mut Genome) {
    let mut rng = rand::thread_rng();
    let mut out_node = *genome
        .nodes
        .values()
        .filter(|n| match n.node_type {
            NodeType::Input => false,
            NodeType::Bias => false,
            _ => true,
        })
        .map(|n| n.node_id)
        .collect::<Vec<_>>()
        .choose(&mut rng)
        .unwrap();
    let mut in_node = *genome
        .nodes
        .values()
        .filter(|n| n.node_id != out_node)
        .map(|n| n.node_id)
        .collect::<Vec<_>>()
        .choose(&mut rng)
        .unwrap();
    if genome.connections.iter().any(|c| c.in_node == in_node && c.out_node == out_node) {
        return
    }
    let weight = rng.gen_range(-4.0, 4.0);
    genome.connections.push(ConnectionGene {
        innovation_number: genome.innovation_counter.update(|n| n + 1),
        in_node,
        out_node,
        weight,
        enabled: true,
    });
}

fn swap_enablement_connection(genome: &mut Genome) {
    let mut rng = rand::thread_rng();
    let connections = &mut genome.connections;

    let choice = connections.choose_mut(&mut rng).unwrap();
    choice.enabled = !choice.enabled;
}

fn split_connection(genome: &mut Genome) {
    let mut rng = rand::thread_rng();
    let connections = &mut genome.connections;
    let nodes = &mut genome.nodes;

    let in_node;
    let out_node;
    let weight;
    {
        let choice = connections.choose_mut(&mut rng).unwrap();
        in_node = choice.in_node;
        out_node = choice.out_node;
        weight = choice.weight;
        choice.enabled = false;
    }
    let new_node_id = genome.node_counter.update(|n| n + 1);
    nodes.insert(
        new_node_id,
        NodeGene {
            node_id: new_node_id,
            node_type: NodeType::Hidden,
        },
    );

    connections.push(ConnectionGene {
        innovation_number: genome.innovation_counter.update(|n| n + 1),
        in_node: in_node,
        out_node: new_node_id,
        weight: 1.0,
        enabled: true,
    });
    connections.push(ConnectionGene {
        innovation_number: genome.innovation_counter.update(|n| n + 1),
        in_node: new_node_id,
        out_node: out_node,
        weight: weight,
        enabled: true,
    });
}

fn crossover(a: &Genome, b: &Genome) -> Genome {
    let mut rng = rand::thread_rng();
    let a_connections: HashMap<_, _> = a
        .connections
        .iter()
        .map(|c| (c.innovation_number, c))
        .collect();
    let b_connections: HashMap<_, _> = b
        .connections
        .iter()
        .map(|c| (c.innovation_number, c))
        .collect();
    let matching_numbers: Vec<_> = a_connections
        .keys()
        .filter(|k| b_connections.contains_key(k))
        .collect();
    let disjoint_numbers: Vec<_> = a_connections
        .keys()
        .filter(|k| !b_connections.contains_key(k))
        .collect();

    let connections: Vec<_> = matching_numbers
        .iter()
        .map(|n| {
            let mut choice = if rng.gen::<f32>() >= 0.5 {
                *a_connections[n]
            } else {
                *b_connections[n]
            };
            if !a_connections[n].enabled || !b_connections[n].enabled {
                if rng.gen::<f32>() < 0.75 {
                    choice.enabled = false;
                } else {
                    choice.enabled = true;
                }
            }
            choice
        })
        .chain(disjoint_numbers.iter().map(|n| *a_connections[n]))
        .collect();

    let nodes: HashMap<_, _> = connections
        .iter()
        .map(|c| (c.in_node, a.nodes[&c.in_node]))
        .chain(
            connections
                .iter()
                .map(|c| (c.out_node, a.nodes[&c.out_node])),
        )
        .collect();

    Genome {
        genome_counter: a.genome_counter.clone(),
        genome_id: a.genome_counter.update(|n| n + 1),
        node_counter: a.node_counter.clone(),
        innovation_counter: a.innovation_counter.clone(),
        nodes,
        connections,
    }
}

fn compatibility_metric(a: &Genome, b: &Genome) -> f32 {
    let c1 = 1.0;
    let c2 = 1.0;
    let c3 = 0.4;
    let largest_genome = a.connections.len().max(b.connections.len());
    let n = if largest_genome > 20 {
        largest_genome as f32
    } else {
        1.0
    };

    let a_connections: HashMap<_, _> = a
        .connections
        .iter()
        .map(|c| (c.innovation_number, c))
        .collect();
    let b_connections: HashMap<_, _> = b
        .connections
        .iter()
        .map(|c| (c.innovation_number, c))
        .collect();

    let b_min_innovation = b
        .connections
        .iter()
        .map(|c| c.innovation_number)
        .min()
        .unwrap();
    let b_max_innovation = b
        .connections
        .iter()
        .map(|c| c.innovation_number)
        .max()
        .unwrap();

    let disjoint_numbers: Vec<_> = a_connections
        .keys()
        .filter(|k| {
            !b_connections.contains_key(k) && **k >= b_min_innovation && **k <= b_max_innovation
        })
        .collect();
    let excess_numbers: Vec<_> = a_connections
        .keys()
        .filter(|k| {
            !b_connections.contains_key(k) && (**k < b_min_innovation || **k > b_max_innovation)
        })
        .collect();
    let matching_numbers: Vec<_> = a_connections
        .keys()
        .filter(|k| b_connections.contains_key(k))
        .collect();
    let w = matching_numbers
        .iter()
        .map(|n| (a_connections[n].weight - b_connections[n].weight).abs())
        .sum::<f32>()
        / matching_numbers.len() as f32;

    (c1 * excess_numbers.len() as f32) / n + (c2 * disjoint_numbers.len() as f32) / n + c3 * w
}

fn speciate(
    type_genomes: &Vec<(usize, f32, Genome)>,
    mut genomes: Vec<Genome>,
) -> Vec<(usize, f32, Vec<Genome>)> {
    let threshold = 3.0;
    let mut type_genomes = type_genomes.clone();
    let mut species: Vec<(usize, f32, Vec<Genome>)> = type_genomes
        .iter()
        .map(|(a, b, _)| (*a, *b, vec![]))
        .collect();

    for genome in genomes.drain(..) {
        let mut matched = None;
        for (i, (_, _, tg)) in type_genomes.iter().enumerate() {
            let compatibility = compatibility_metric(&genome, tg);
            if compatibility < threshold {
                matched = Some(i);
                break;
            }
        }
        if let Some(i) = matched {
            species[i].2.push(genome);
        } else {
            let new_type = genome.clone();
            species.push((0, 0.0, vec![genome]));
            type_genomes.push((0, 0.0, new_type));
        }
    }

    species.drain(..).filter(|s| s.2.len() != 0).collect()
}

pub fn next_generation(
    fitness: &HashMap<usize, f32>,
    previous_generation: &Vec<(usize, f32, Vec<Genome>)>,
) -> Vec<(usize, f32, Vec<Genome>)> {
    let mut rng = rand::thread_rng();
    let pop_size = fitness.len();
    let mut previous_generation = previous_generation.clone();

    let species_fitness: Vec<f32> = previous_generation
        .iter()
        .map(|(_, _, s)| s.iter().map(|g| fitness[&g.genome_id]).sum::<f32>() / s.len() as f32)
        .collect();
    let species_fitness: Vec<f32> = species_fitness
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let (last_increase, last_value, _) = &previous_generation[i];
            if *last_increase >= 15 && last_value >= v {
                0.0
            } else {
                if last_value < v {
                    previous_generation[i].0 = 0;
                    previous_generation[i].1 = *v;
                } else {
                    previous_generation[i].0 += 0;
                }
                *v
            }
        })
        .collect();

    let children_per_unit =
        (pop_size as f32 * 0.75) / (species_fitness.iter().sum::<f32>() + 0.00000000000001);
    let children_per_species: Vec<usize> = species_fitness
        .iter()
        .map(|f| (f * children_per_unit) as usize)
        .collect();
    
    // remove the weak ones
    previous_generation = previous_generation.iter().enumerate().map(|(i, s)| {
        let threshold = species_fitness[i] * 0.3;
        (s.0, s.1, s.2.iter().cloned().filter(|g| fitness[&g.genome_id] >= threshold).collect())
    }).collect();

    //TODO interspecies mating
    let mut next_generation: Vec<Genome> = vec![];
    for (i, c) in children_per_species.iter().enumerate() {
        let mut c = *c;
        if previous_generation[i].2.len() > 1 {
            if c > 5 {
                let mut champ = previous_generation[i].2.iter().max_by_key(|g| (fitness[&g.genome_id]*10000.0) as i32).unwrap().clone();
                c -= 1;
                champ.genome_id = champ.genome_counter.update(|n| n+1);
                next_generation.push(champ);
            }
            for _ in 0..c {
                let mut a = previous_generation[i]
                    .2
                    .choose_weighted(&mut rng, |g| fitness[&g.genome_id])
                    .unwrap();
                let mut b = previous_generation[i].2.choose(&mut rng).unwrap();
                while b.genome_id == a.genome_id {
                    b = previous_generation[i].2.choose(&mut rng).unwrap();
                }
                if fitness[&b.genome_id] > fitness[&a.genome_id] {
                    let c = a;
                    a = b;
                    b = c;
                }
                next_generation.push(maybe_mutate(crossover(a, b)));
            }
        } else if previous_generation[i].2.len() == 1 && c > 0 {
            let g = &previous_generation[i].2[0];
            let mut g = g.clone();
            g.genome_id = g.genome_counter.update(|n| n + 1);
            next_generation.push(maybe_mutate(g));
        }
    }
    let randos = pop_size - next_generation.len();
    for _ in 0..randos {
        let g = previous_generation
            .iter()
            .map(|(_, _, s)| s)
            .flatten()
            .choose(&mut rng)
            .unwrap()
            .clone();
        let mut g = g.clone();
        g.genome_id = g.genome_counter.update(|n| n + 1);
        next_generation.push(maybe_mutate(g));
    }
    let type_genomes: Vec<(usize, f32, Genome)> = previous_generation
        .iter()
        .filter(|s| s.2.len() > 0)
        .map(|(a, b, s)| (*a, *b, s.choose(&mut rng).unwrap().clone()))
        .collect();
    speciate(&type_genomes, next_generation)
}

fn maybe_mutate(mut g: Genome) -> Genome {
    let mut rng = rand::thread_rng();
    if rng.gen::<f32>() < 0.8 {
        for connection in g.connections.iter_mut() {
            if rng.gen::<f32>() < 0.9 {
                connection.weight += rng.gen_range(-0.1, 0.1);
            } else {
                connection.weight = rng.gen_range(-2.0, 2.0);
            }
        }
    }
    if rng.gen::<f32>() < 0.003 {
        split_connection(&mut g);
    }
    if rng.gen::<f32>() < 0.01 {
        swap_enablement_connection(&mut g);
    }
    if rng.gen::<f32>() < 0.005 {
        add_connection(&mut g);
    }
    g
}

pub fn initial_population(
    size: usize,
    inputs: usize,
    outputs: usize,
) -> Vec<(usize, f32, Vec<Genome>)> {
    let mut rng = rand::thread_rng();
    let genome_counter = Rc::new(Cell::new(0));
    let innovation_counter = Rc::new(Cell::new(0));
    let node_counter = Rc::new(Cell::new(0));
    let mut input_nodes: Vec<NodeGene> = (0..inputs)
        .map(|i| NodeGene {
            node_id: node_counter.update(|n| n + 1),
            node_type: NodeType::Input,
        })
        .collect();
    input_nodes.push(NodeGene {
        node_id: node_counter.update(|n| n + 1),
        node_type: NodeType::Bias,
    });
    let mut output_nodes: Vec<NodeGene> = (0..outputs)
        .map(|i| NodeGene {
            node_id: node_counter.update(|n| n + 1),
            node_type: NodeType::Output,
        })
        .collect();
    let mut connections: Vec<ConnectionGene> = vec![];
    for input in 0..input_nodes.len() {
        for output in 0..output_nodes.len() {
            connections.push(ConnectionGene {
                innovation_number: innovation_counter.update(|n| n + 1),
                in_node: input_nodes[input].node_id,
                out_node: output_nodes[output].node_id,
                weight: 0.0,
                enabled: true,
            });
        }
    }
    let mut nodes: HashMap<usize, NodeGene> = input_nodes
        .drain(..)
        .chain(output_nodes.drain(..))
        .map(|n| (n.node_id, n))
        .collect();

    let pop: Vec<Genome> = (0..size)
        .map(|_| {
            let mut connections: Vec<ConnectionGene> = connections.clone();
            for g in &mut connections {
                g.weight = rng.gen_range(-4.0, 4.0)
            }
            Genome {
                genome_id: genome_counter.update(|n| n + 1),
                genome_counter: genome_counter.clone(),
                node_counter: node_counter.clone(),
                innovation_counter: innovation_counter.clone(),
                nodes: nodes.clone(),
                connections,
            }
        })
        .collect();

    speciate(&vec![(0, 0.0, pop[0].clone())], pop)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_and() {
        let mut net = Phenotype {
            bias_node: 0,
            input_nodes: vec![1, 2],
            output_nodes: vec![3],
            nodes: vec![0.0, 0.0, 0.0, 0.0],
            connections: vec![
                (0, 3, -1.4),
                (1, 3, 1.0), // AND 1
                (2, 3, 1.0), // AND 2
            ],
        };
        for (a, b) in &[(true, true), (true, false), (false, true), (false, false)] {
            let a = *a;
            let b = *b;
            let truth = if a && b { 1.0 } else { 0.0 };
            let output = net.evaluate(&[if a { 1.0 } else { 0.0 }, if b { 1.0 } else { 0.0 }]);
            assert!(output[0].round() == truth);
        }
    }

    #[test]
    fn test_xor() {
        let mut net = Phenotype {
            bias_node: 0,
            input_nodes: vec![1, 2],
            output_nodes: vec![3],
            nodes: vec![0.0, 0.0, 0.0, 0.0, 0.0],
            connections: vec![(0, 3, -5.0), (1, 3, 20.0), (2, 3, 10.0)],
        };
        for (a, b) in &[(true, true), (true, false), (false, true), (false, false)] {
            let a = *a;
            let b = *b;
            let truth = if (a || b) && !(a && b) { 1.0 } else { 0.0 };
            let output = net.evaluate(&[if a { 1.0 } else { 0.0 }, if b { 1.0 } else { 0.0 }]);
            eprintln!("{}", output[0]);
            assert!(output[0].round() == truth);
        }
    }
}
