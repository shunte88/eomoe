# eomoe - Expert One-Way Context MoE LLM

A memory-safe, massively scalable implementation of a Mixture of Experts (MoE) Large Language Model system with efficient GPU memory management.

## Features

- **Memory-Safe Rust Implementation**: unsafe confined to the CUDA FFI calls in `memory.rs`
- **GPU Memory Management**: Efficient VRAM/CUDA context swapping with expert loading/unloading
- **MoE Architecture**: Complete Mixture of Experts model with expert weights and routing
- **REST API Server**: HTTP endpoints for inference, health checks, and model information
- **CLI Interface**: Command-line tools for installation, serving, and checking
- **LRU Caching**: Automatic expert caching and swapping based on usage patterns

## Architecture

The system is designed around:

1. **MemoryManager**: Manages GPU memory allocation and expert swapping between VRAM and system memory
2. **ExpertCache**: LRU cache implementation for MoE experts
3. **MoEModel**: Core model structure with expert weights and parameters
4. **HTTP Server**: REST API endpoints for model inference and management
5. **CLI Interface**: Command-line tools for system operations

## Components

### Memory Management
- GPU memory allocation/deallocation using CUDA API calls
- System memory caching for experts
- Automatic swapping between GPU VRAM and system memory
- Memory statistics and monitoring

### Model Loading
- GGUF file format parsing
- Expert loading and unloading
- Model configuration management

### HTTP Server
- REST API endpoints:
  - `GET /health` - Health check
  - `GET /info` - Model information
  - `POST /inference` - Model inference with tokenization

### CLI Commands
- `eomoe install` - Install required dependencies
- `eomoe serve` - Start the HTTP server
- `eomoe check` - System health check

## Usage

```bash
# Build the project
cargo build --release

# Start the server (flags only)
./target/release/eomoe serve --model /path/to/model.gguf --port 8080

# Start from a config file; any flag overrides the file
./target/release/eomoe --config config.example.yaml serve --port 9000
EOMOE_CONFIG_PATH=config.example.yaml ./target/release/eomoe serve

# Check system health
./target/release/eomoe check

# Install dependencies
./target/release/eomoe install
```

## Dependencies

- Rust 1.70+
- CUDA runtime libraries (for GPU operations). `.cargo/config.toml` points the linker
  at `/usr/local/cuda/targets/x86_64-linux/lib`; set `CUDA_LIBRARY_PATH` to override.
- cuBLAS library (for GPU linear algebra operations)

## Implementation Notes

This implementation provides a complete, memory-safe MoE architecture that can handle large models with limited GPU memory through efficient caching and swapping mechanisms. All components are designed with thread safety in mind, using Arc<Mutex<MoEModel>> patterns for safe concurrent access.