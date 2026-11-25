//! Configuration management for eomoe

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Main configuration structure for eomoe
#[derive(Debug, Deserialize, Serialize)]
pub struct EomoeConfig {
    /// Model configuration
    pub model: Option<ModelConfig>,
    /// Server configuration
    pub server: Option<ServerConfig>,
    /// GPU configuration
    pub gpu: Option<GpuConfig>,
    /// Logging configuration
    pub logging: Option<LoggingConfig>,
}

/// Model configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ModelConfig {
    pub path: String,
    pub cache_size_mib: u32,
    pub num_experts: usize,
    pub expert_dim: usize,
    pub num_layers: usize,
}

/// Server configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ServerConfig {
    pub port: u16,
    pub host: String,
    pub max_connections: u32,
}

/// GPU configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GpuConfig {
    pub device_id: u32,
    pub memory_limit_mib: u32,
}

/// Logging configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LoggingConfig {
    pub level: String,
    pub file: String,
}

impl EomoeConfig {
    /// Load configuration from a YAML file
    pub fn load_from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: EomoeConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Merge with CLI arguments, giving precedence to CLI values
    pub fn merge_with_cli(&self, cli_args: &crate::cli::ServeArgs) -> ServeArgsWithConfig {
        ServeArgsWithConfig {
            model: cli_args.model.clone(),
            port: cli_args.port,
            cache_size: cli_args.cache_size,
            device: cli_args.device,
            // Use config values if not overridden by CLI
            model_config: self.model.clone(),
            server_config: self.server.clone(),
            gpu_config: self.gpu.clone(),
        }
    }
}

/// Serve arguments with configuration
#[derive(Debug)]
pub struct ServeArgsWithConfig {
    pub model: std::path::PathBuf,
    pub port: u16,
    pub cache_size: u32,
    pub device: u32,
    pub model_config: Option<ModelConfig>,
    pub server_config: Option<ServerConfig>,
    pub gpu_config: Option<GpuConfig>,
}