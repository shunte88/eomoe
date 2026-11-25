//! Simplified HTTP server for eomoe with REST API endpoints

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
use tokio::sync::Mutex;

use crate::model::MoEModel;

/// Server state containing model
#[derive(Debug, Clone)]
pub struct ServerState {
    pub model: Arc<Mutex<MoEModel>>,
}

/// API error response
#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// Model inference request
#[derive(Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

/// Model inference response
#[derive(Serialize)]
pub struct InferenceResponse {
    pub text: String,
    pub tokens: Vec<String>,
    pub usage: UsageStats,
}

/// Usage statistics
#[derive(Serialize)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Health check endpoint
pub async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "healthy" }))
}

/// Model info endpoint
pub async fn model_info(State(state): State<ServerState>) -> impl IntoResponse {
    // Get model info from the model state
    let model = state.model.lock().await;
    Json(serde_json::json!({
        "model": "MoE LLM",
        "status": "ready",
        "expert_count": model.expert_count(),
        "config": model.config()
    }))
}

/// Inference endpoint
pub async fn inference(
    State(state): State<ServerState>,
    Json(request): Json<InferenceRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    // In a real implementation, this would:
    // - Process the prompt using the MoE model
    // - Route to appropriate experts
    // - Handle GPU memory management
    // - Return the generated text

    // For now, return a mock response with proper structure
    let response = InferenceResponse {
        text: format!("Mock response to: {}", request.prompt),
        tokens: vec!["token1".to_string(), "token2".to_string(), "token3".to_string()],
        usage: UsageStats {
            prompt_tokens: request.prompt.chars().count(),
            completion_tokens: 3,
            total_tokens: request.prompt.chars().count() + 3,
        }
    };

    Ok(Json(response))
}

/// Start the HTTP server
pub async fn start_server(
    model: Arc<Mutex<MoEModel>>,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create server state
    let state = ServerState {
        model,
    };

    // Define routes
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/info", get(model_info))
        .route("/inference", post(inference))
        .with_state(state);

    // Bind to the specified port
    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).await?;

    println!("Server starting on port {}", port);

    // Start the server
    axum::serve(listener, app).await?;

    Ok(())
}