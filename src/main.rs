//! eomoe - Expert One-Way Context MoE LLM
//!
//! A memory-safe, massively scalable implementation of Mixture of Experts
//! Large Language Model system with efficient GPU memory management.

use clap::Parser;
use std::sync::Arc;
use tokio::sync::Mutex;

mod cli;
mod memory;
mod model;
mod cache;
mod server_simple;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let cli_args = cli::CliArgs::parse();

    match &cli_args.command {
        cli::Command::Install(install_args) => {
            cli::execute_install(install_args).await?;
        }
        cli::Command::Serve(serve_args) => {
            // Create model, memory manager, and cache
            // In a real implementation, this would be more sophisticated
            let model_config = model::ModelConfig {
                model_path: serve_args.model.to_string_lossy().to_string(),
                cache_size_mib: serve_args.cache_size,
                num_experts: 128,
                expert_dim: 1024,
                num_layers: 24,
            };

            let model = model::MoEModel::new(model_config).unwrap();
            // Start the server with the model
            server_simple::start_server(
                Arc::new(Mutex::new(model)),
                serve_args.port,
            ).await?;
        }
        cli::Command::Check(check_args) => {
            cli::execute_check(check_args).await?;
        }
    }

    Ok(())
}