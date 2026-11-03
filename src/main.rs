use clap::Parser;

use icode::cli::Cli;

fn main() {
    let cli = Cli::parse();
    if let Err(error) = icode::cli::execute(cli) {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}
