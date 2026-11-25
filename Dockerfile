# Multi-stage build for eomoe
FROM rust:1.70 as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    libcuda-dev \
    libcudart-dev \
    libcublas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Cargo files
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src/ src/
COPY README.md .

# Build the application
RUN cargo build --release

# Final stage
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libcuda1 \
    libcudart10 \
    libcublas10 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy the binary from builder stage
COPY --from=builder /app/target/release/eomoe .

# Create directories for config and models
RUN mkdir -p /app/config /app/models

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["./eomoe", "serve", "--model", "/app/models/model.gguf", "--port", "8080"]