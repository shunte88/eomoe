//! HTTP server implementation for eomoe MoE model
//!
//! Implements REST API endpoints for MoE model inference, health checks,
//! and model information.

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpListener;

use crate::model::MoEModel;
use crate::memory::MemoryManager;
use crate::cache::ExpertCache;

/// Server state containing model and memory manager
#[derive(Debug, Clone)]
pub struct ServerState {
    pub model: Arc<MoEModel>,
    pub memory_manager: Arc<MemoryManager>,
    pub expert_cache: Arc<ExpertCache>,
}

/// API error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> axum::response::Response {
        let body = Json(self);
        (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
    }
}

/// Health check endpoint
pub async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "eomoe-moe-server"
    }))
}

/// Model info endpoint
pub async fn model_info(State(state): State<ServerState>) -> impl IntoResponse {
    let config = state.model.config();
    Json(serde_json::json!({
        "model": config.model_path,
        "num_experts": config.num_experts,
        "expert_dim": config.expert_dim,
        "num_layers": config.num_layers,
        "cache_size_mib": config.cache_size_mib,
    }))
}

/// Model inference endpoint
pub async fn model_inference(
    State(state): State<ServerState>,
    Json(payload): Json<InferenceRequest>,
) -> impl IntoResponse {
    // In a real implementation, this would:
    // - Load the requested experts into GPU memory if not already cached
    // - Perform MoE inference with the loaded experts
    // - Return the inference results

    // For now, we'll simulate inference
    let response = InferenceResponse {
        prompt: payload.prompt,
        response: "This is a simulated MoE inference response".to_string(),
        expert_used: payload.expert_id.unwrap_or(0),
    };

    Json(response)
}

/// Inference request payload
#[derive(Debug, Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub expert_id: Option<usize>,
}

/// Inference response payload
#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    pub prompt: String,
    pub response: String,
    pub expert_used: usize,
}

/// Start the HTTP server
pub async fn start_server(
    model: Arc<MoEModel>,
    memory_manager: Arc<MemoryManager>,
    expert_cache: Arc<ExpertCache>,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = ServerState {
        model,
        memory_manager,
        expert_cache,
    };

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/info", get(model_info))
        .route("/infer", post(model_inference))
        .with_state(state);

    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    println!("Server listening on http://0.0.0.0:{}", port);

    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check() {
        let response = health_check().await;
        assert_eq!(response.0.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_model_info() {
        // This would require a mock model to test
        // For now, we just verify the function compiles
        assert!(true);
    }
}