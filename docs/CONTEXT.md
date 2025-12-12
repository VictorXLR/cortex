# Cortex Runtime - Project Context

> **Purpose of this document:** Provide full context for human collaborators and AI subagents working on this project. Read this first before making changes.

## The Big Picture

We're building a **social gaming/messaging platform** with these elements:

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE PLATFORM VISION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📱 Messaging App (core)                                        │
│      + 🎮 Virtual Worlds (office, campus, custom)               │
│      + 🤖 AI NPCs that live, learn, remember                    │
│      + 🎭 ARG storylines (mysteries, events)                    │
│      + 👥 Multiplayer (real players + AI)                       │
│      + 📍 Location-based social (Pokemon Go style)              │
│      + 📺 Streaming integration (Twitch/YouTube)                │
│                                                                  │
│  Think: Discord + The Sims + Pokemon Go + ARG + Twitch          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why Cortex Matters

Cortex is the **runtime that powers the AI NPCs**. Each NPC needs to:
- Have persistent memory (remember players, events, relationships)
- Maintain state across sessions (personality, skills, goals)
- Execute behaviors (daily routines, role-playing, reactions)
- Feel alive (not just a chatbot - they have their own lives)

## Related Projects

| Project | Location | Purpose |
|---------|----------|---------|
| **cortex** | `/Users/loopy/Developer/ai/cortex` | Native AI runtime - memory, state, inference |
| **neural_assembly** | `/Users/loopy/Developer/ai/neural_assembly` | Research: teaching transformers CPU-like execution |

### How They Connect

```
neural_assembly (research)          cortex (runtime)
├── Differentiable PC          →    Agent behavior execution
├── Addressable Memory         →    Memory subsystem
├── Call Stack                 →    Context/role switching
├── CPU-like instructions      →    High-level life actions
└── Learned execution traces   →    NPC decision making
```

The neural_assembly concepts may eventually inform how cortex agents "think" - but for now, cortex focuses on practical runtime infrastructure.

## Current State of Cortex

### Architecture Overview

```
cortex/
├── src/
│   ├── lib.rs              # Core types, Message, Role enums
│   ├── main.rs             # CLI application
│   ├── runtime.rs          # Cortex struct - main runtime
│   ├── session.rs          # Persistent session wrapper
│   ├── config.rs           # Configuration structures
│   ├── memory/
│   │   ├── mod.rs          # Memory subsystem (store + retrieve)
│   │   └── vector.rs       # Vector store with cosine similarity
│   ├── inference/
│   │   ├── mod.rs          # InferenceEngine trait, chat templates
│   │   └── llama.rs        # llama.cpp backend implementation
│   └── state/
│       ├── mod.rs          # StateManager for persistence
│       └── checkpoint.rs   # Checkpoint and Branch structs
└── Cargo.toml
```

### What Works

| Component | Status | Notes |
|-----------|--------|-------|
| Chat/Generation | ✅ Working | Streaming responses via llama.cpp |
| Session Persistence | ✅ Working | Auto-saves to ~/.local/share/cortex/sessions/ |
| Message History | ✅ Working | Tracks conversation with roles |
| Memory Storage | ⚠️ Partial | Structure works, but search is broken |
| Checkpoint API | ⚠️ Partial | Can save/load metadata, not actual KV cache |
| CLI Interface | ✅ Working | chat, generate, sessions commands |
| Configuration | ✅ Working | Model path, GPU layers, context size, etc. |

### What's Broken

| Issue | Location | Impact | Priority |
|-------|----------|--------|----------|
| **Embeddings return zeros** | `src/inference/llama.rs:164` | Memory search always fails | 🔴 Critical |
| **KV cache not serialized** | `src/inference/llama.rs:279` | Can't truly restore state | 🔴 Critical |
| **No API server** | N/A | CLI only, can't serve platform | 🟡 High |
| **Single agent only** | N/A | Can't run multiple NPCs | 🟡 High |
| **No event system** | N/A | Can't react to world events | 🟠 Medium |

### Code Locations for Key Issues

**Embedding stub (returns zeros):**
```rust
// src/inference/llama.rs:164
// TODO: Use proper sentence embedding model or pooling strategy
let _tokens = self.tokenize(text, false)?;
Ok(vec![0.0; self.embedding_dim()])  // <-- BROKEN: always zeros
```

**KV cache stub (not implemented):**
```rust
// src/inference/llama.rs:279
Ok(KvCacheState {
    data: vec![],  // TODO: Implement actual cache extraction
    n_tokens: self.tokens_in_context,
    model_id: self.model_id.clone(),
})
```

## Target Architecture (v1)

```
┌─────────────────────────────────────────────────────────────────┐
│                      CORTEX v1 RUNTIME                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    API LAYER                             │    │
│  │  HTTP/WebSocket server for platform integration          │    │
│  │  - POST /agent/{id}/message  (chat with NPC)            │    │
│  │  - GET  /agent/{id}/state    (inspect NPC state)        │    │
│  │  - POST /agent/{id}/event    (world event notification) │    │
│  │  - WS   /agent/{id}/stream   (streaming responses)      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  AGENT MANAGER                           │    │
│  │  Manages multiple agent instances                        │    │
│  │  - Agent pool with lifecycle management                  │    │
│  │  - Shared model inference (one model, many agents)       │    │
│  │  - Per-agent state isolation                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │   AGENT 1    │ │   AGENT 2    │ │   AGENT N    │             │
│  │   (Maya)     │ │   (Bob)      │ │   (...)      │             │
│  │              │ │              │ │              │             │
│  │ - Session    │ │ - Session    │ │ - Session    │             │
│  │ - Memory     │ │ - Memory     │ │ - Memory     │             │
│  │ - State      │ │ - State      │ │ - State      │             │
│  │ - Personality│ │ - Personality│ │ - Personality│             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 INFERENCE ENGINE                         │    │
│  │  Shared llama.cpp instance                               │    │
│  │  - Model loaded once                                     │    │
│  │  - Request queue for fair scheduling                     │    │
│  │  - Working embeddings (sentence-transformers or pooling) │    │
│  │  - KV cache management per agent                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 PERSISTENCE LAYER                        │    │
│  │  - Agent state (memory, checkpoints)                     │    │
│  │  - Session history                                       │    │
│  │  - Vector store (with real embeddings)                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Decisions Made

1. **Rust stays** - Performance matters for multi-agent runtime
2. **llama.cpp backend** - Local inference, no API dependencies
3. **No external DBs for v1** - File-based persistence (SQLite maybe later)
4. **HTTP + WebSocket API** - Standard integration with platform
5. **Shared model, isolated state** - Efficient multi-agent support

## Key Decisions Needed

- [ ] Embedding strategy: Use separate embedding model or pool from LLM?
- [ ] API framework: axum, actix-web, or warp?
- [ ] Agent scheduling: Round-robin, priority queue, or async all?
- [ ] State format: Keep bincode or switch to something debuggable?

## For AI Subagents

When working on this codebase:

1. **Read this file first** - Understand the vision and current state
2. **Check ROADMAP.md** - See what's currently being worked on
3. **Don't over-engineer** - We're targeting a weekend v1
4. **Keep it working** - Small, tested increments
5. **Update docs** - If you learn something, document it

### Common Tasks

**Adding a new API endpoint:**
1. Add route in `src/api/mod.rs` (once created)
2. Add handler in appropriate module
3. Update this doc if it changes the architecture

**Fixing the embedding issue:**
1. Look at `src/inference/llama.rs:164`
2. Options: mean pooling, separate model, or external service
3. Test with `src/memory/vector.rs` similarity search

**Adding a new agent capability:**
1. Consider if it belongs in Agent, Session, or Runtime
2. Keep state serializable (serde)
3. Think about multi-agent implications

## Contact / Collaboration

This is a hyper-collaborative project. When in doubt:
- Document your assumptions
- Ask clarifying questions
- Make small PRs/changes
- Keep the runtime working
