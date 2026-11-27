//! Command-line interface: the one-stop launcher (serve, check, codex, claude, install),
//! with the same commands, settings and client wiring as the reference `local-code-llm`.

use clap::{Args, Parser, Subcommand};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::Command as Proc;

pub const MODEL_NAME: &str = "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf";
const MODEL_URL: &str = "https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/dff23eb494e11eb6508c4964dc909cf97923aabb/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf";
const MODEL_SHA256: &str = "fadc3e5f8d42bf7e894a785b05082e47daee4df26680389817e2093056f088ad";

#[derive(Parser)]
#[clap(name = "eomoe", version = concat!("v", env!("CARGO_PKG_VERSION"), " built ", env!("EOMOE_BUILT")), about = "Local MoE LLM server for Claude Code and Codex")]
pub struct CliArgs {
    /// YAML configuration file (CLI flags and env vars override its values)
    #[clap(long, env = "EOMOE_CONFIG_PATH", global = true)]
    pub config: Option<PathBuf>,

    #[clap(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)] // parsed once
pub enum Command {
    /// Download the model and install the Codex and Claude Code clients
    Install,
    /// Start the model server on loopback. Leave it running.
    Serve(ServeArgs),
    /// Test streamed inference over Responses and Messages
    Check(ClientArgs),
    /// Launch Codex CLI on the local Responses endpoint
    Codex(ClientArgs),
    /// Launch Claude Code on the local Messages endpoint
    Claude(ClientArgs),
    /// List third-party components and their licenses
    About,
}

#[derive(Args)]
pub struct ServeArgs {
    /// Model GGUF [config: model.path, default: <home>/models/<Qwen3-Coder Q4_K_M>]
    #[clap(long, env = "MODEL_FILE")]
    pub model: Option<PathBuf>,
    /// Address to bind [config: server.host, default: 127.0.0.1]
    #[clap(long)]
    pub host: Option<String>,
    /// Port [config: server.port, default: 8080]
    #[clap(long, env = "PORT")]
    pub port: Option<u16>,
    /// Context size in tokens [config: model.ctx_size, default: 32768]
    #[clap(long, env = "CTX_SIZE")]
    pub ctx_size: Option<usize>,
    /// VRAM expert cache in MiB [config: model.cache_mib, default: all free VRAM]
    #[clap(long, env = "MOE_CACHE_MIB")]
    pub cache_mib: Option<usize>,
    /// Experts per token, fewer than the model's for speed at some quality cost [config: model.moe_experts, default: the model's]
    #[clap(long = "moeexperts", env = "MOE_EXPERTS")]
    pub moe_experts: Option<usize>,
    /// Requests served at once; they share the context's KV pool [config: server.parallel, default: 4]
    #[clap(long, env = "PARALLEL")]
    pub parallel: Option<usize>,
    /// Host-RAM prompt cache in MiB: displaced prompts' KV, restored when they return; 0 off, -1 no limit [config: server.cache_ram, default: 0 until its gate runs]
    #[clap(long, env = "LLAMA_ARG_CACHE_RAM", allow_hyphen_values = true)]
    pub cache_ram: Option<i64>,
    /// Largest request body in MiB; larger gets 413 [config: server.max_request_mb, default: 32]
    #[clap(long, env = "MAX_REQUEST_MB")]
    pub max_request_mb: Option<usize>,
    /// Requests that may wait beyond --parallel; more get 503 [config: server.queue_limit, default: no limit]
    #[clap(long, env = "QUEUE_LIMIT")]
    pub queue_limit: Option<usize>,
    /// Seconds a request may take, queueing included; then generation stops (finish "length") [config: server.req_timeout, default: no limit]
    #[clap(long, env = "REQ_TIMEOUT")]
    pub req_timeout: Option<u64>,
    /// Model name served to clients [config: server.alias, default: local-code]
    #[clap(long)]
    pub alias: Option<String>,
    /// API keys, comma-separated; requests then need one (Bearer or x-api-key) [config: server.api_key]
    #[clap(long, env = "LLAMA_API_KEY")]
    pub api_key: Option<String>,
    /// File of API keys, one per line, `#` comments [config: server.api_key_file]
    #[clap(long, env = "LLAMA_ARG_API_KEY_FILE")]
    pub api_key_file: Option<PathBuf>,
    /// PEM private key: serve HTTPS (with --ssl-cert-file) [config: server.ssl_key_file]
    #[clap(long, env = "LLAMA_ARG_SSL_KEY_FILE")]
    pub ssl_key_file: Option<PathBuf>,
    /// PEM certificate chain for HTTPS [config: server.ssl_cert_file]
    #[clap(long, env = "LLAMA_ARG_SSL_CERT_FILE")]
    pub ssl_cert_file: Option<PathBuf>,
}

#[derive(Args)]
pub struct ClientArgs {
    /// Server port on this machine (ignored with --url)
    #[clap(long, env = "PORT", default_value_t = 8080)]
    pub port: u16,
    /// Server base URL, for a remote or HTTPS server, e.g. https://gpu-box:8080
    #[clap(long, env = "EOMOE_URL")]
    pub url: Option<String>,
    /// API key, if the server was started with --api-key
    #[clap(long, env = "EOMOE_API_KEY")]
    pub api_key: Option<String>,
    /// Context size the clients are told about
    #[clap(long, env = "CTX_SIZE", default_value_t = crate::config::DEFAULT_CTX)]
    pub ctx_size: usize,
    /// Arguments passed through to the client
    #[clap(trailing_var_arg = true, allow_hyphen_values = true)]
    pub args: Vec<String>,
}

impl ClientArgs {
    fn base(&self) -> String {
        self.url.as_deref().map_or_else(|| format!("http://127.0.0.1:{}", self.port), |u| u.trim_end_matches('/').to_owned())
    }
}

/// Where eomoe keeps the model, the clients and their config: $EOMOE_HOME, default
/// ~/.local/share/eomoe. Missing model / clients fall back to a reference install
/// (~/.local/share/local-code-llm), so an existing setup works without re-downloading.
pub fn home() -> PathBuf {
    std::env::var_os("EOMOE_HOME").map(PathBuf::from).unwrap_or_else(|| data_dir().join("eomoe"))
}

fn data_dir() -> PathBuf {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local/share"))
}

/// `rel` under eomoe's home, else under the reference install if it exists there.
fn installed(rel: &str) -> PathBuf {
    let own = home().join(rel);
    let reference = data_dir().join("local-code-llm").join(rel);
    if !own.exists() && reference.exists() { reference } else { own }
}

pub fn default_model() -> PathBuf {
    installed(&format!("models/{MODEL_NAME}"))
}

fn client(name: &str) -> Result<PathBuf, String> {
    let p = installed(&format!("clients/node_modules/.bin/{name}"));
    if p.exists() { Ok(p) } else { Err(format!("{} not found: run `eomoe install`", p.display())) }
}

fn run(cmd: &mut Proc) -> Result<(), String> {
    let st = cmd.status().map_err(|e| format!("{cmd:?}: {e}"))?;
    if st.success() { Ok(()) } else { Err(format!("{cmd:?} failed: {st}")) }
}

pub fn install() -> Result<(), String> {
    let h = home();
    for d in ["models", "clients", "codex", "claude"] {
        std::fs::create_dir_all(h.join(d)).map_err(|e| e.to_string())?;
    }
    let model = default_model();
    if !model.exists() {
        let part = model.with_extension("gguf.part");
        // One request per invocation; rerunning resumes the partial file.
        run(Proc::new("curl").args(["--fail", "--location", "--continue-at", "-", "--output"]).arg(&part).arg(MODEL_URL))?;
        let out = Proc::new("sha256sum").arg(&part).output().map_err(|e| e.to_string())?;
        if !String::from_utf8_lossy(&out.stdout).starts_with(MODEL_SHA256) {
            return Err(format!("model checksum mismatch: remove {} and rerun install", part.display()));
        }
        std::fs::rename(&part, &model).map_err(|e| e.to_string())?;
    }
    if client("codex").is_err() || client("claude").is_err() {
        run(Proc::new("npm").args(["install", "--prefix"]).arg(h.join("clients")).args(["--no-audit", "--no-fund", "@openai/codex", "@anthropic-ai/claude-code"]))?;
    }
    println!("model: {}\nStart the server with `eomoe serve`, then run check, codex or claude from another terminal.", model.display());
    Ok(())
}

/// POST `body` to `base` + `path` and return the `data: ` payloads of the streamed response.
pub fn post_sse(base: &str, key: Option<&str>, path: &str, body: &serde_json::Value) -> Result<Vec<String>, String> {
    let mut r = ureq::post(&format!("{base}{path}")).timeout(std::time::Duration::from_secs(600)).set("Content-Type", "application/json");
    if let Some(k) = key {
        r = r.set("Authorization", &format!("Bearer {k}"));
    }
    let resp = r.send_string(&body.to_string()).map_err(|e| match e {
        ureq::Error::Status(c, r) => format!("{c}: {}", r.into_string().unwrap_or_default()),
        e => format!("{e}. Check that serve is running and ready."),
    })?;
    Ok(BufReader::new(resp.into_reader()).lines().map_while(Result::ok).filter_map(|l| l.strip_prefix("data: ").map(str::to_owned)).collect())
}

/// The reference `check`: both endpoints must stream non-empty text and complete.
pub fn check(a: &ClientArgs) -> Result<(), String> {
    let prompt = "What does len([1, 2]) return in Python? Reply briefly.";
    let cases = [
        ("/v1/responses", serde_json::json!({"input": prompt, "max_output_tokens": 64, "store": false}), "response.output_text.delta", "response.completed"),
        ("/v1/messages", serde_json::json!({"messages": [{"role": "user", "content": prompt}], "max_tokens": 64}), "content_block_delta", "message_stop"),
    ];
    for (path, mut body, delta, done) in cases {
        body["model"] = "local-code".into();
        body["stream"] = true.into();
        let (mut text, mut completed) = (String::new(), false);
        for d in post_sse(&a.base(), a.api_key.as_deref(), path, &body).map_err(|e| format!("{path}: {e}"))? {
            if d == "[DONE]" {
                break;
            }
            let ev: serde_json::Value = serde_json::from_str(&d).map_err(|e| format!("{path}: {e}"))?;
            if ev["type"] == delta {
                let v = &ev["delta"];
                text += v.as_str().or(v["text"].as_str()).unwrap_or("");
            }
            completed |= ev["type"] == done;
        }
        if !completed || text.trim().is_empty() {
            return Err(format!("{path}: no complete streamed text response."));
        }
        println!("{path}: {}", text.trim());
    }
    Ok(())
}

/// Replace this process with Codex, pointed at the local Responses endpoint.
pub fn codex(a: &ClientArgs) -> Result<(), String> {
    let base = a.base();
    let mut c = Proc::new(client("codex")?);
    if let Some(k) = &a.api_key {
        c.env("EOMOE_API_KEY", k).arg("-c").arg("model_providers.local_llama.env_key=\"EOMOE_API_KEY\"");
    }
    c.env("CODEX_HOME", home().join("codex")).args(["-m", "local-code"]);
    for kv in [
        "model_provider=\"local_llama\"".to_owned(),
        "model_providers.local_llama.name=\"Local eomoe\"".to_owned(),
        format!("model_providers.local_llama.base_url=\"{base}/v1\""),
        "model_providers.local_llama.wire_api=\"responses\"".to_owned(),
        "model_providers.local_llama.supports_websockets=false".to_owned(),
        "web_search=\"disabled\"".to_owned(),
        format!("model_context_window={}", a.ctx_size),
    ] {
        c.arg("-c").arg(kv);
    }
    exec(c.args(&a.args))
}

/// Replace this process with Claude Code, pointed at the local Messages endpoint.
pub fn claude(a: &ClientArgs) -> Result<(), String> {
    let mut c = Proc::new(client("claude")?);
    for k in ["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"] {
        c.env_remove(k);
    }
    c.env("CLAUDE_CONFIG_DIR", home().join("claude"))
        .env("ANTHROPIC_BASE_URL", a.base())
        .env("ANTHROPIC_AUTH_TOKEN", a.api_key.as_deref().unwrap_or("local"));
    for k in ["ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"] {
        c.env(k, "local-code");
    }
    c.env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1").env("ENABLE_TOOL_SEARCH", "false");
    // "local-code" is unknown to Claude Code, which would assume a 1M window and never compact
    // before the server's context overflows.
    c.env("CLAUDE_CODE_MAX_CONTEXT_TOKENS", a.ctx_size.to_string());
    exec(c.args(["--model", "local-code"]).args(&a.args))
}

fn exec(c: &mut Proc) -> Result<(), String> {
    use std::os::unix::process::CommandExt;
    let e = c.exec();
    Err(format!("{:?}: {e}", c.get_program()))
}

/// `eomoe about`: third-party components and their licenses.
pub const ABOUT: &str = include_str!("about.txt");

#[cfg(test)]
mod tests {
    #[test]
    fn about_lists_every_dependency() {
        // Stale after a dependency change? Regenerate: sh tools/gen_about.sh
        let toml = include_str!("../Cargo.toml");
        let deps = toml.split("[dependencies]").nth(1).unwrap().split("\n[").next().unwrap();
        for name in deps.lines().filter_map(|l| l.split_once(" = ")).map(|(n, _)| n.trim()) {
            assert!(super::ABOUT.lines().any(|l| l.starts_with(&format!("{name} v"))), "{name} missing from src/about.txt");
        }
        for c in ["ggml", "Qwen3-Coder", "Apache-2.0"] {
            assert!(super::ABOUT.contains(c), "{c}");
        }
    }
}
