use dynamic_weighted_index::DynamicWeightedIndex;
use itertools::Itertools;
use pcg_rand::Pcg64;
use rand::{distributions::Distribution, Rng, SeedableRng};
use std::time::Instant;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "non_linear_preferential_attachment",
    about = "Generates an edge list using non-linear preferential attachment"
)]
struct Opt {
    #[structopt(short = "i", long)]
    initial_nodes: Option<usize>,

    #[structopt(short = "s", long)]
    seed_value: Option<u64>,

    #[structopt(short = "n", long)]
    nodes: usize,

    #[structopt(short = "d", long, default_value = "1")]
    initial_degree: usize,

    #[structopt(short = "e", long, default_value = "1.0")]
    exponent: f64,

    #[structopt(short = "c", long, default_value = "0.0")]
    offset: f64,

    #[structopt(short = "p", long)]
    simple_graph: bool,

    #[structopt(short = "r", long)]
    report_degree_distribution: bool,
}

fn get_and_check_options() -> Opt {
    let mut opt = Opt::from_args();

    assert!(opt.initial_degree >= 1);
    if opt.initial_nodes.is_none() {
        opt.initial_nodes = Some(opt.initial_degree * 10);
    }
    assert!(opt.initial_nodes.unwrap() >= opt.initial_degree);

    assert!(opt.exponent >= 0.0);
    assert!(opt.offset >= 0.0);

    opt
}

fn degree_distribution(degrees: &[usize]) -> Vec<(usize, usize)> {
    let mut counts = degrees.iter().copied().counts().into_iter().collect_vec();
    counts.sort_unstable();
    counts
}

fn execute_preferential_attachment(opt: Opt, rng: &mut impl Rng) {
    let total_nodes = opt.initial_nodes.unwrap() + opt.nodes;

    let mut degrees = vec![0usize; total_nodes];
    let mut dyn_index = DynamicWeightedIndex::new(total_nodes);

    let cached_weights = (0..100)
        .into_iter()
        .map(|d| opt.offset + (d as f64).powf(opt.exponent))
        .collect_vec();

    // we use a macro to update the degree to "trick" the borrow checker
    macro_rules! set_degree {
        ($node:expr, $degree:expr) => {{
            let u = $node;
            let d = $degree;

            degrees[u] = d;

            let weight = if cached_weights.len() > d {
                cached_weights[d as usize]
            } else {
                opt.offset + (d as f64).powf(opt.exponent)
            };

            dyn_index.set_weight(u, weight);
        }};
    }

    // build initial circle
    for u in 0..opt.initial_nodes.unwrap() {
        set_degree!(u, 2);
    }

    // run preferential attachment
    let mut hosts = vec![opt.initial_degree];

    for u in opt.initial_nodes.unwrap()..total_nodes {
        // sample neighbors
        for h in &mut hosts {
            *h = dyn_index.sample(rng).unwrap();
            if opt.simple_graph && opt.initial_degree > 1 {
                dyn_index.set_weight(*h, 0.0);
            };
        }

        // update neighbors
        for &h in &hosts {
            set_degree!(h, degrees[h] + 1);
        }

        set_degree!(u, opt.initial_degree);
    }

    if opt.report_degree_distribution {
        let degree_distr = degree_distribution(&degrees);
        println!(
            "{}",
            degree_distr
                .iter()
                .map(|&(d, n)| format!("{:>10}, {:>10}", d, n))
                .join("\n")
        );
    }
}

fn main() {
    let opt = get_and_check_options();

    let mut rng = if let Some(seed_value) = opt.seed_value {
        Pcg64::seed_from_u64(seed_value)
    } else {
        Pcg64::from_entropy()
    };

    let start = Instant::now();
    execute_preferential_attachment(opt, &mut rng);
    eprintln!(
        "Total runtime: {} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
}
