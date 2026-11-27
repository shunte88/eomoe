use clap::Parser;
use eomoe::cli::{self, Command};
use eomoe::{config::EomoeConfig, gguf::Gguf, gpu::GpuModel, server, template::ChatTemplate, tokenizer::Tokenizer};

fn main() {
    if let Err(e) = run() {
        eprintln!("eomoe: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = cli::CliArgs::parse();
    let config = match &args.config {
        Some(p) => EomoeConfig::load(p)?,
        None => EomoeConfig::default(),
    };
    // RUST_LOG still wins over the config file.
    let level = config.logging.level.as_deref().unwrap_or("info");
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level)).init();

    match &args.command {
        Command::Install => cli::install(),
        Command::About => {
            print!("{}", cli::ABOUT);
            Ok(())
        }
        Command::Check(a) => cli::check(a),
        Command::Codex(a) => cli::codex(a),
        Command::Claude(a) => cli::claude(a),
        Command::Serve(a) => {
            let s = config.resolve_serve(a)?;
            log::info!("model={} ctx={} cache={:?} MiB parallel={} cache_ram={}MiB api_keys={} https={}", s.model.display(), s.ctx_size, s.cache_mib, s.parallel, s.cache_ram >> 20, s.api_keys.len(), s.tls.is_some());
            if !(1..=eomoe::gpu::MAX_SEQ).contains(&s.parallel) {
                return Err(format!("--parallel must be 1..={}", eomoe::gpu::MAX_SEQ));
            }
            let g = Gguf::open(&s.model)?;
            let (tok, tpl) = (Tokenizer::from_gguf(&g)?, ChatTemplate::from_gguf(&g)?);
            let (path, ctx, cache, par) = (s.model.clone(), s.ctx_size, s.cache_mib.unwrap_or(usize::MAX >> 20), s.parallel);
            // The engine is built on, and stays on, the server's model thread.
            let app = server::App::new(tok, tpl, s.alias.clone(), server::ollama::describe(&g, &s.model), s.cache_ram, move || {
                let mut m = GpuModel::with_cache(Gguf::open(&path)?, ctx, cache)?;
                m.set_n_seq(par);
                Ok(m)
            })?;
            let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
            rt.block_on(server::serve(app, &s.host, s.port, s.api_keys, s.tls))
        }
    }
}
