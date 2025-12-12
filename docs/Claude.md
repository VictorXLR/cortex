The Real Problem

  Every AI application rebuilds the same thing:

  Your App
      ↓
  [Orchestration code you write]
      ↓
  ┌─────────┬─────────┬─────────┬─────────┐
  │ OpenAI  │Pinecone │ Redis   │ Your DB │
  │ API     │ API     │ Cache   │         │
  └─────────┴─────────┴─────────┴─────────┘
      ↓           ↓         ↓         ↓
   Network    Network   Network   Network

  Four services. Four API keys. Four billing accounts. JSON serialization everywhere. State
  scattered across all of them. You write 200 lines of glue code before doing anything
  useful.

  AI has no runtime.

  JavaScript has V8/Node/Bun. Python has CPython. Java has JVM. These provide memory
  management, I/O, execution primitives.

  AI applications have... nothing. We're writing assembly, calling external services for
  everything.

  ---
  What Is A Runtime?

  A runtime provides primitives your code assumes exist:

  | Runtime  | Memory        | State                 | I/O               | Execution
           |
  |----------|---------------|-----------------------|-------------------|-------------------
  ---------|
  | V8/Node  | Heap, GC      | Call stack, closures  | fs, net, http     | Event loop
           |
  | JVM      | Heap, GC      | Stack frames, threads | java.io, java.net | Bytecode
  interpreter       |
  | AI today | ❌ External DB | ❌ DIY                 | ❌ HTTP APIs       | ❌
  Orchestration frameworks |

  Cortex provides the missing primitives:

  | Cortex    | What it provides                     |
  |-----------|--------------------------------------|
  | Memory    | Vector store, key-value, persistence |
  | State     | Checkpoint, restore, branch, merge   |
  | I/O       | Native tool execution                |
  | Execution | Inference engine, context management |

  ---
  The Insight: State Is The Missing Primitive

  The real insight isn't "faster inference" or "fewer dependencies."

  It's that AI state doesn't exist as a concept.

  In traditional programming, the runtime manages state:
  - Stack frames (function calls)
  - Heap (objects)
  - Process state (can fork, checkpoint)

  In AI, "state" is scattered:
  - KV cache (inside the model, inaccessible)
  - Conversation history (in your database)
  - Long-term memory (in Pinecone)
  - Tool state (wherever you put it)

  And none of it is addressable or checkpointable.

  Cortex makes AI state a first-class primitive:

  // Checkpoint entire state: KV cache + memory + context
  let snap = ctx.checkpoint()?;

  // Try something
  let a = ctx.chat("What if we go left?")?;

  // Restore, try something else
  ctx.restore(&snap)?;
  let b = ctx.chat("What if we go right?")?;

  // Compare a vs b, pick best path

  This enables:
  - Speculative execution - try multiple approaches
  - Tree-of-thought - explore reasoning branches
  - Session continuity - resume exactly where you left off
  - Debugging - replay from any point
  - What-if analysis - branch and explore

  ---
  The API (What Developers See)

  Basic

  let ctx = Cortex::load("llama-3-8b.gguf")?;
  let response = ctx.chat("Hello!")?;

  Memory

  // Write (auto-embeds strings)
  ctx.memory.write("user:prefs", "Likes jazz, hates meetings")?;

  // Search
  let relevant = ctx.memory.search("What music?", 5)?;

  // Memory auto-injected into chat when relevant
  let response = ctx.chat("Recommend me something")?;
  // ^ Knows user likes jazz

  State

  let snap = ctx.checkpoint()?;
  // ... do stuff ...
  ctx.restore(&snap)?;

  // Or branch for parallel exploration
  let branch = ctx.branch()?;
  branch.chat("Explore this path")?;
  ctx.chat("Explore that path")?;

  Sessions (High-Level)

  // Auto-persists everything
  let session = Cortex::session("model.gguf", "user_123")?;
  session.chat("Hello!")?;

  // Later, new process:
  let session = Cortex::session("model.gguf", "user_123")?;
  // Fully restored - memory, context, everything
  session.chat("What were we discussing?")?;

  ---
  Differentiation

  |              | LangChain            | Ollama           | vLLM               | Cortex
    |
  |--------------|----------------------|------------------|--------------------|------------
  --|
  | What it is   | Orchestration        | Server           | Production serving | Runtime
    |
  | Memory       | External             | None             | None               | Built-in
    |
  | State        | DIY                  | None             | None               | Built-in
    |
  | Checkpoint   | No                   | No               | No                 | Yes
    |
  | Branch       | No                   | No               | No                 | Yes
    |
  | Dependencies | Pinecone, Redis, etc | None but limited | Production infra   | None
    |
  | Target       | Glue code            | Easy local       | Scale              |
  Applications |

  Cortex is not:
  - An orchestration framework (LangChain)
  - A model server (Ollama)
  - A production serving system (vLLM)
  - An inference library (llama.cpp)

  Cortex is: A runtime. Like Node is for JavaScript.

  ---
  The Pitch

  Cortex: The AI Runtime

  Memory and state as primitives, not infrastructure.

  let ctx = Cortex::load("model.gguf")?;
  ctx.memory.write("fact", "The sky is blue")?;
  let snap = ctx.checkpoint()?;
  let response = ctx.chat("What color is the sky?")?;

  No Pinecone. No Redis. No LangChain. No orchestration.
  One binary. Just run.

  ---
  Positioning Options

  1. "The AI Runtime" - The foundational layer
  2. "SQLite for AI" - Embedded, zero-config, local-first
  3. "Bun for AI" - Fast, batteries-included, developer-focused

  I lean toward "SQLite for AI" because:
  - SQLite's value prop is clear: embedded database, zero config, just works
  - Same idea: embedded AI runtime, zero config, just works
  - SQLite became the default for local data; Cortex becomes default for local AI

  ---
  MVP Scope

  Week 1:
  1. Load GGUF via llama-cpp
  2. generate() / chat() API
  3. In-memory vector store
  4. checkpoint() / restore()

  Week 2:
  5. Disk persistence (memory + state)
  6. Session API (auto-persist)
  7. CLI tool

  Week 3:
  8. Tool registration
  9. Auto memory retrieval
  10. Branching
  11. Docs + examples

  ---
  File Structure

  cortex/
  ├── src/
  │   ├── lib.rs           # Public API: Cortex, Session
  │   ├── runtime.rs       # Core runtime
  │   ├── memory/
  │   │   ├── mod.rs       # Memory trait + impl
  │   │   ├── vector.rs    # Vector store (HNSW)
  │   │   └── persist.rs   # Disk persistence
  │   ├── state/
  │   │   ├── mod.rs       # State management
  │   │   ├── checkpoint.rs
  │   │   └── branch.rs
  │   ├── inference/
  │   │   ├── mod.rs       # Inference engine trait
  │   │   └── llama.rs     # llama-cpp backend
  │   ├── session.rs       # High-level session API
  │   └── config.rs
  ├── examples/
  │   ├── basic.rs
  │   ├── memory.rs
  │   └── branching.rs
  └── README.md

  ---
  Goals

  1. Zero infrastructure - One binary, no external services
  2. State as primitive - Checkpoint/branch/restore are core ops
  3. Memory as primitive - Built-in, not bolted-on
  4. Native performance - Rust, no Python, no serialization tax
  5. Developer experience - Simple API, just works

  ---
  Ready to write the code?
