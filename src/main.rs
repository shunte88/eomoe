use clap::Parser;
use eomoe::{cli, config::EomoeConfig, model, server};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = cli::CliArgs::parse();

    let config = match &args.config {
        Some(path) => EomoeConfig::load(path)?,
        None => EomoeConfig::default(),
    };

    // RUST_LOG still wins over the config file.
    let level = config.logging.level.as_deref().unwrap_or("info");
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level)).init();

    match &args.command {
        cli::Command::Install(a) => cli::execute_install(a).await?,
        cli::Command::Check(a) => cli::execute_check(a).await?,
        cli::Command::Serve(a) => {
            let s = config.resolve_serve(a)?;
            log::info!(
                "model={} cache={} MiB device={}",
                s.model.model_path,
                s.model.cache_size_mib,
                s.device
            );
            let model = model::MoEModel::new(s.model)?;
            server::start_server(Arc::new(Mutex::new(model)), &s.host, s.port).await?;
        }
    }

    Ok(())
}
