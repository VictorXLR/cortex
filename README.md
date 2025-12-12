# Cortex

A Rust-based AI inference and memory system built with Candle. Provides local LLM inference with persistent memory and state management.

## Features

- **Local LLM Inference**: GGUF model support via Candle framework
- **Semantic Memory**: Vector storage with similarity search and optional embedding models
- **Session Management**: Persistent chat sessions with state checkpointing
- **State Checkpointing**: Save and restore conversation states
- **CLI Interface**: Interactive chat and single-shot generation modes

## Installation

```bash
git clone https://github.com/VictorXLR/cortex.git
cd cortex
cargo build --release
```

## Usage

### Interactive Chat
```bash
# Basic chat (requires GGUF model file)
cortex chat --model path/to/model.gguf

# Chat with session persistence
cortex chat --model path/to/model.gguf --session my-session

# Chat with semantic memory (downloads embedding model on first use)
cortex chat --model path/to/model.gguf --memory

# Custom parameters
cortex chat --model path/to/model.gguf --temperature 0.8 --max-tokens 2048
```

### Single Generation
```bash
cortex generate --model path/to/model.gguf "Explain quantum computing"
```

### Session Management
```bash
# List all sessions
cortex sessions

# Delete a session
cortex delete-session my-session
```

### Model Information
```bash
cortex info --model path/to/model.gguf
```

## Memory System

The memory system provides semantic search capabilities:

- **Vector Storage**: Efficient similarity search with configurable thresholds
- **Automatic Embeddings**: Downloads and uses embedding models when `--memory` flag is used
- **Persistent Storage**: Memory persists across sessions
- **Configurable Limits**: Set maximum entries and similarity thresholds

## Models

Place GGUF format models in the `models/` directory. The system supports various quantized models compatible with Candle.

## Configuration

The system uses sensible defaults but can be configured via:
- Command line arguments
- Configuration files (see `src/config.rs`)
- Environment variables

## Documentation

- [Context](docs/CONTEXT.md) - Project context and architecture
- [Contributing](docs/CONTRIBUTING.md) - Development guidelines
- [Roadmap](docs/ROADMAP.md) - Future plans
- [Status](docs/STATUS.md) - Current implementation status
- [Claude Notes](docs/Claude.md) - Development notes

## Architecture

- **Runtime**: Core execution environment with memory and state primitives
- **Inference**: Pluggable text generation backends (Candle, stub engines)
- **Memory**: Vector storage with similarity search
- **State**: Checkpoint and session management
- **Config**: Centralized configuration system

## License

MIT