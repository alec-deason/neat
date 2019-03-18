extern crate neat;
extern crate rand;
extern crate serde_json;
extern crate pancurses;
extern crate rayon;

use rayon::prelude::*;

use pancurses::{initscr, endwin, noecho, curs_set};


use std::env;

use std::fs::File;
use std::io::prelude::*;

use rand::prelude::*;

use rand::prelude::SliceRandom;

use std::collections::HashMap;

use neat::{initial_population, next_generation, Genome, Phenotype};

#[derive(Copy, Clone, Debug)]
enum TileType {
    Floor,
    Wall,
    ClosedChest(usize),
    OpenChest(usize),
}

impl From<TileType> for f32 {
    fn from(tile: TileType) -> Self {
        match tile {
            TileType::Floor => 0.0 / 3.0,
            TileType::Wall => 1.0 / 3.0,
            TileType::ClosedChest(_) => 2.0 / 3.0,
            TileType::OpenChest(_) => 1.0 / 3.0,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum Action {
    Forward,
    Back,
    TurnLeft,
    TurnRight,
    OpenChest,
    LootChest,
}

impl From<f32> for Action {
    fn from(action: f32) -> Self {
        let action = ((action * 6.0) % 6.0).round() as usize;
        match action {
            0 => Action::Forward,
            1 => Action::Back,
            2 => Action::TurnLeft,
            3 => Action::TurnRight,
            4 => Action::OpenChest,
            _ => Action::LootChest,
        }
    }
}

struct Creature {
    ai: Phenotype,
    x: usize,
    y: usize,
    facing: usize,
}

impl Creature {
    fn from_phenotype(phenotype: &Phenotype, x: usize, y: usize) -> Creature {
        Creature {
            ai: phenotype.clone(),
            x,
            y,
            facing: 0,
        }
    }

    fn step(&mut self, input_tiles: &[TileType]) -> Action {
        let inputs:Vec<f32> = input_tiles.iter().map(|t| (*t).into()).collect();
        let output = self.ai.evaluate(&inputs);
        output[0].into()
    }
}

struct Dungeon {
    tiles: Vec<Vec<TileType>>,
    adventurer: Creature,
    score: usize,
}

impl Dungeon {
    fn window_size() -> (usize, usize) {
        (6, 6)
    }

    fn input_size() -> usize {
        let (wx, wy) = Self::window_size();
        (wx / 2) * 2 * (wy / 2) * 2
    }

    fn new(phenotype: &Phenotype, width: usize, height: usize, chest_count: usize, score_per_chest: usize, seed: u64) -> Dungeon {
        let mut rng: StdRng = StdRng::seed_from_u64(seed);
        let mut tiles = vec![vec![TileType::Floor; height]; width];
        for x in &[0, width-1] {
            for y in 0..height-1 {
                tiles[*x][y] = TileType::Wall;
            }
        }
        for y in &[0, height-1] {
            for x in 0..width-1 {
                tiles[x][*y] = TileType::Wall;
            }
        }

        let mut chest_count = chest_count;
        while chest_count > 0 {
            let x = rng.gen_range(0, width-1);
            let y = rng.gen_range(0, height-1);
            match tiles[x][y] {
                TileType::Floor => {
                    tiles[x][y] = TileType::ClosedChest(score_per_chest);
                    chest_count -= 1;
                },
                _ => ()
            }
        }


        let adventurer = Creature::from_phenotype(phenotype, 10, 10);

        Dungeon{
            tiles,
            adventurer,
            score: 0
        }
    }

    fn step(&mut self) -> bool {
        let x = self.adventurer.x;
        let y = self.adventurer.y;
        let mut fx = x;
        let mut fy = y;
        match self.adventurer.facing {
            0 => fy -= 1, // Forward
            1 => fx += 1, // Right
            2 => fy += 1, // Back
            3 => fx -= 1, // Left
            _ => (),
        }

        let mut input_tiles = Vec::new();
        let (mut wx, mut wy) = Self::window_size();
        wx /= 2;
        wy /= 2;
        for dx in -(wx as i32) .. wx as i32 {
            for dy in -(wy as i32) .. wy as i32 {
                let mut dx = dx;
                let mut dy = dy;
                match self.adventurer.facing {
                    0 => (), // Forward
                    1 => { // Right
                        dx = dy;
                        dy = -dx;
                    },
                    2 => { // Back
                        dy = -dy;
                    },
                    3 => {// Left
                        dx = -dy;
                        dy = dx;
                    },
                    _ => (),
                }
                let x = x as i32 + dx;
                let y = y as i32 + dy;
                input_tiles.push(if x >= 0 && x < self.tiles.len() as i32 && y >= 0 && y < self.tiles[0].len() as i32 {
                    self.tiles[x as usize][y as usize]
                } else {
                    TileType::Wall
                });
            }
        };

        let action = self.adventurer.step(&input_tiles);
        match action {
            Action::Forward => {
                let front_tile;
                if fx >= 0 && fx < self.tiles.len() && fy >= 0 && fy < self.tiles[0].len() {
                    front_tile = self.tiles[fx][fy];
                } else {
                    front_tile = TileType::Wall;
                }
                match front_tile {
                    TileType::Floor | TileType::OpenChest(_) | TileType::ClosedChest(_) => {
                        self.adventurer.x = fx;
                        self.adventurer.y = fy;
                    },
                    _ => ()
                }
            },
            Action::Back => {
                let mut bx = x;
                let mut by = y;
                match self.adventurer.facing {
                    0 => by -= 1, // Forward
                    1 => bx += 1, // Right
                    2 => by += 1, // Back
                    3 => bx -= 1, // Left
                    _ => ()
                }
                let back_tile;
                if bx >= 0 && bx < self.tiles.len() && by >= 0 && by < self.tiles[0].len() {
                    back_tile = self.tiles[bx][by];
                } else {
                    back_tile = TileType::Wall;
                }
                match back_tile {
                    TileType::Floor | TileType::OpenChest(_) | TileType::ClosedChest(_) => {
                        self.adventurer.x = bx;
                        self.adventurer.y = by;
                    },
                    _ => ()
                }
            },
            Action::TurnRight => {
                self.adventurer.facing = (self.adventurer.facing + 1) % 4;
            },
            Action::TurnLeft => {
                if self.adventurer.facing == 0 {
                    self.adventurer.facing = 4;
                } else {
                    self.adventurer.facing -= 1;
                }
            },
            Action::OpenChest => {
                match self.tiles[self.adventurer.x][self.adventurer.y] {
                    TileType::ClosedChest(score) => {
                        self.tiles[self.adventurer.x][self.adventurer.y] = TileType::OpenChest(score);
                    },
                    _ => ()
                }
            },
            Action::LootChest => {
                match self.tiles[self.adventurer.x][self.adventurer.y] {
                    TileType::OpenChest(score) => {
                        self.score += score;
                        self.tiles[self.adventurer.x][self.adventurer.y] = TileType::Floor;
                    },
                    _ => ()
                }
            }
        }

        for col in &self.tiles {
            for tile in col {
                match tile {
                    TileType::ClosedChest(_) => return false,
                    TileType::OpenChest(_) => return false,
                    _ => ()
                }
            }
        }
        true
    }


}

fn main() {
    let width = 30;
    let height = 30;
    let chests = ((width * height) as f32 * 0.2) as usize;

    let score_per_chest = 10;
    let argv:Vec<_> = env::args().collect();
    if argv.len() > 1 {
        let mut f = File::open(&argv[1]).unwrap();
        let mut buffer = String::new();
        f.read_to_string(&mut buffer).unwrap();
        let phenotype:Phenotype = serde_json::from_str(&buffer).unwrap();
        let seed = rand::thread_rng().gen::<u64>();
        let mut dungeon = Dungeon::new(&phenotype, width, height, chests, score_per_chest, seed);
        let window = initscr();
        noecho();
        curs_set(0);
        loop {
            for x in 0..dungeon.tiles.len() {
                for y in 0..dungeon.tiles[0].len() {
                    window.mvprintw(y as i32, x as i32,
                        match dungeon.tiles[x][y] {
                            TileType::Wall => "#",
                            TileType::OpenChest(_) => "o",
                            TileType::ClosedChest(_) => "c",
                            _ => ".",
                        }
                    );

                }
            }
            window.mvprintw(dungeon.adventurer.y as i32, dungeon.adventurer.x as i32, "@");
            window.refresh();
            window.getch();
            dungeon.step();
        }
        endwin();
        return
    }

    let mut rng = rand::thread_rng();
    let mut pop = initial_population(150, Dungeon::input_size(), 1);
    let mut generation = 0;



    loop {
        let phenotypes:Vec<_> = pop.iter().map(|s| &s.2).flatten().map(|genome| (genome.genome_id, Phenotype::from_genome(genome))).collect();
        let mut fitness:HashMap<usize, f32> = phenotypes.par_iter().map(|(genome_id, phenotype)| {
            let mut e = 0.0;
            for test_id in 0..10 {
                let mut dungeon = Dungeon::new(phenotype, width, height, chests, score_per_chest, test_id as u64);
                for _ in 0..100 {
                    let done = dungeon.step();
                    if done {
                        e += 100.0;
                        break;
                    } else {
                        //e -= 0.1;
                    }
                }
                e = (e + dungeon.score as f32).max(0.0);
            }
            e /= ((chests * score_per_chest) as f32 + 100.0) * 10.0;
            (*genome_id, e)
        }).collect();
        let fitness: HashMap<usize, f32> = fitness.iter().map(|(k, v)| (*k, v.powf(2.0))).collect();
        if generation % 100 == 0 {
            let (most_fit, _) = fitness.iter().max_by_key(|(k,v)| (*v * 10000.0) as i32).unwrap();
            for s in &pop {
                for genome in &s.2 {
                    if genome.genome_id == *most_fit {
                        let phenotype = Phenotype::from_genome(genome);
                        let json = serde_json::to_string(&phenotype);
                        let mut file = File::create("/tmp/test.json").unwrap();
						write!(&mut file, "{}", json.unwrap());

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
