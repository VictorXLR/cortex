# Cortex - Live Status

> Quick reference for current project state. Update this as you work.

## Current Sprint: Weekend v1

**Goal:** Multi-agent runtime with working memory and HTTP API

```
Overall Progress: [██░░░░░░░░] 15%
```

## Task Board

### 🔴 In Progress
*Nothing currently in progress*

### 🟡 Up Next
| Task | Owner | Notes |
|------|-------|-------|
| Fix embeddings (llama.rs:164) | - | Critical blocker |
| Fix KV cache persistence | - | Critical blocker |

### ✅ Completed
| Task | Completed By | Date |
|------|--------------|------|
| Project documentation | Claude | 2024-12-10 |
| CONTEXT.md created | Claude | 2024-12-10 |
| ROADMAP.md created | Claude | 2024-12-10 |
| CONTRIBUTING.md created | Claude | 2024-12-10 |

### 🚫 Blocked
*Nothing currently blocked*

---

## Quick Health Check

| Component | Status | Last Verified |
|-----------|--------|---------------|
| Build | ⚠️ Unknown | - |
| Tests | ⚠️ Unknown | - |
| CLI Chat | ⚠️ Unknown | - |
| Memory Search | ❌ Broken (zeros) | - |
| Checkpoints | ⚠️ Partial | - |
| API | ❌ Not Started | - |

---

## Current Blockers

1. **Embeddings return zeros** - Memory search doesn't work
   - Location: `src/inference/llama.rs:164`
   - Impact: Core feature broken

2. **KV cache not serialized** - Can't restore LLM state
   - Location: `src/inference/llama.rs:279`
   - Impact: Checkpoints incomplete

---

## Environment

```bash
# To start working:
cd /Users/loopy/Developer/ai/cortex
cargo build

# Run CLI:
cargo run -- --help

# Run tests:
cargo test
```

---

## Recent Changes

```
2024-12-10 - Added project documentation (CONTEXT, ROADMAP, CONTRIBUTING, STATUS)
```

---

## Notes

*Add quick notes here:*

---

**Last Updated:** 2024-12-10 by Claude
