# Cortex v1 Roadmap - Weekend Sprint

> **Goal:** Functional multi-agent runtime with working memory and API by end of weekend

## Current Status

```
[█░░░░░░░░░] 10% - Foundations exist, core features broken
```

**Last Updated:** 2024-12-10
**Target:** v1.0 by Sunday night

---

## Weekend Plan

### Day 1 (Saturday): Fix Fundamentals

**Morning: Embeddings (Critical)**
```
[ ] Fix embedding generation in llama.rs
    - Option A: Mean pooling over token embeddings
    - Option B: Use last token embedding
    - Option C: Integrate separate embedding model (e5, bge, etc.)

[ ] Test memory search actually works
    - Store 10 test memories
    - Query with semantic similarity
    - Verify relevant results returned
```

**Afternoon: KV Cache Persistence**
```
[ ] Implement actual KV cache extraction
    - Use llama_state_get_size / llama_state_get_data
    - Serialize to checkpoint

[ ] Implement KV cache restoration
    - Use llama_state_set_data
    - Verify generation continues correctly

[ ] Test checkpoint/restore cycle
    - Save mid-conversation
    - Restore and continue
    - Verify coherent responses
```

**Evening: Multi-Agent Foundation**
```
[ ] Create Agent struct (wraps Session + personality)
    src/agent/mod.rs
    - agent_id: String
    - name: String
    - personality: AgentPersonality
    - session: Session
    - state: AgentState

[ ] Create AgentManager
    src/agent/manager.rs
    - agents: HashMap<AgentId, Agent>
    - create_agent()
    - get_agent()
    - list_agents()
    - remove_agent()
```

---

### Day 2 (Sunday): API & Integration

**Morning: HTTP API**
```
[ ] Add axum dependency
[ ] Create basic server structure
    src/api/mod.rs
    src/api/routes.rs
    src/api/handlers.rs

[ ] Implement core endpoints:
    POST /agents              - Create new agent
    GET  /agents              - List all agents
    GET  /agents/{id}         - Get agent info
    POST /agents/{id}/chat    - Send message, get response
    GET  /agents/{id}/memory  - Query agent's memory
    POST /agents/{id}/remember - Store memory
```

**Afternoon: WebSocket Streaming**
```
[ ] Add WebSocket support
    WS /agents/{id}/stream

[ ] Implement streaming chat
    - Send message via WS
    - Receive tokens as they generate
    - Handle connection lifecycle
```

**Evening: Polish & Test**
```
[ ] Integration test: Create agent via API
[ ] Integration test: Multi-turn conversation
[ ] Integration test: Memory recall
[ ] Basic error handling and logging
[ ] Update README with API docs
[ ] Tag v1.0.0
```

---

## File Structure After Weekend

```
cortex/
├── src/
│   ├── lib.rs                 # Exports
│   ├── main.rs                # CLI + server startup
│   │
│   ├── runtime.rs             # Core Cortex runtime
│   ├── session.rs             # Session wrapper
│   ├── config.rs              # Configuration
│   │
│   ├── agent/                 # NEW
│   │   ├── mod.rs             # Agent struct
│   │   ├── manager.rs         # AgentManager
│   │   └── personality.rs     # AgentPersonality, traits
│   │
│   ├── api/                   # NEW
│   │   ├── mod.rs             # Server setup
│   │   ├── routes.rs          # Route definitions
│   │   ├── handlers.rs        # Request handlers
│   │   └── ws.rs              # WebSocket handlers
│   │
│   ├── memory/
│   │   ├── mod.rs             # Memory subsystem
│   │   └── vector.rs          # Vector store (FIXED)
│   │
│   ├── inference/
│   │   ├── mod.rs             # InferenceEngine trait
│   │   └── llama.rs           # llama.cpp (FIXED)
│   │
│   └── state/
│       ├── mod.rs             # StateManager
│       └── checkpoint.rs      # Checkpoints (FIXED)
│
├── tests/                     # NEW
│   ├── api_tests.rs
│   ├── agent_tests.rs
│   └── memory_tests.rs
│
├── CONTEXT.md                 # Project context (this doc)
├── ROADMAP.md                 # This file
├── README.md                  # Updated with API docs
└── Cargo.toml                 # Updated deps
```

---

## API Spec (Target)

### Create Agent
```http
POST /agents
Content-Type: application/json

{
  "name": "Maya",
  "personality": {
    "traits": ["friendly", "curious", "organized"],
    "background": "Office manager at Nexus Corp, loves spreadsheets",
    "speaking_style": "professional but warm"
  },
  "model_config": {
    "temperature": 0.7,
    "max_tokens": 512
  }
}

Response: 201 Created
{
  "id": "agent_abc123",
  "name": "Maya",
  "created_at": "2024-12-10T10:00:00Z"
}
```

### Chat with Agent
```http
POST /agents/agent_abc123/chat
Content-Type: application/json

{
  "message": "Hey Maya, how's your day going?",
  "player_id": "player_xyz",  // optional, for relationship tracking
  "context": {                 // optional, world state
    "time": "morning",
    "location": "office_lobby"
  }
}

Response: 200 OK
{
  "response": "Oh, hi! Pretty busy actually - quarterly reports are due...",
  "agent_state": {
    "mood": "slightly_stressed",
    "activity": "working"
  }
}
```

### Stream Chat (WebSocket)
```
WS /agents/agent_abc123/stream

-> {"type": "message", "content": "Hey Maya!"}
<- {"type": "token", "content": "Oh"}
<- {"type": "token", "content": ", hi"}
<- {"type": "token", "content": "!"}
<- {"type": "done", "full_response": "Oh, hi!"}
```

---

## Dependencies to Add

```toml
# Cargo.toml additions for v1

# API
axum = "0.7"
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace"] }

# WebSocket
tokio-tungstenite = "0.21"

# Serialization
serde_json = "1.0"  # already have serde

# Utilities
chrono = { version = "0.4", features = ["serde"] }
```

---

## Success Criteria for v1

- [ ] Can create multiple agents via API
- [ ] Agents remember conversations across restarts
- [ ] Memory search returns semantically relevant results
- [ ] Can stream responses via WebSocket
- [ ] Checkpoint/restore actually preserves LLM state
- [ ] Clean shutdown saves all agent states
- [ ] Documentation updated

---

## Post-Weekend (v1.1+)

These are explicitly **out of scope** for the weekend:

- World simulation integration
- ARG event system
- Relationship tracking between agents
- Location-based features
- Streaming platform integration
- Production deployment (Docker, etc.)
- Authentication/authorization
- Rate limiting
- Metrics/monitoring

We'll tackle these after v1 is solid.

---

## Notes / Blockers

*Add notes here as you work:*

```
[DATE] [WHO] - Note
─────────────────────────────────────────
```

---

## Quick Commands

```bash
# Build
cargo build --release

# Run CLI
cargo run -- chat --model /path/to/model.gguf

# Run server (after API is implemented)
cargo run -- serve --port 3000

# Run tests
cargo test

# Run with logging
RUST_LOG=debug cargo run -- serve
```
