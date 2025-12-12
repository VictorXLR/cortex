# Contributing to Cortex

## Collaboration Model

This project uses **hyper-collaborative development** - multiple humans and AI agents working together rapidly. Here's how to contribute effectively.

## Before You Start

1. **Read CONTEXT.md** - Understand the vision and architecture
2. **Check ROADMAP.md** - See current priorities and status
3. **Look at open issues** - Don't duplicate work

## Making Changes

### Small Changes (< 50 lines)
Just make the change. Update docs if needed.

### Medium Changes (50-200 lines)
1. Comment in ROADMAP.md what you're working on
2. Make the change
3. Test it
4. Update ROADMAP.md with status

### Large Changes (> 200 lines)
1. Document your plan first
2. Break into smaller pieces if possible
3. Update architecture docs if changing structure

## Code Style

```rust
// Good: Clear, simple, documented
/// Creates a new agent with the given personality.
pub fn create_agent(name: &str, personality: AgentPersonality) -> Result<Agent> {
    // Validate name
    if name.is_empty() {
        return Err(CortexError::InvalidAgentName);
    }

    // Create agent with default session
    let session = Session::new()?;
    Ok(Agent {
        id: Uuid::new_v4().to_string(),
        name: name.to_string(),
        personality,
        session,
    })
}

// Bad: Clever, over-engineered, uncommented
pub fn create_agent(n: &str, p: AgentPersonality) -> Result<Agent> {
    (!n.is_empty()).then(|| Agent {
        id: Uuid::new_v4().to_string(),
        name: n.into(),
        personality: p,
        session: Session::new().ok()?,
    }).ok_or(CortexError::InvalidAgentName)
}
```

### Principles
- **Clarity over cleverness**
- **Simple over complex**
- **Working over perfect**
- **Documented over mysterious**

## Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_memory_search

# Run with output
cargo test -- --nocapture
```

### What to Test
- Public API functions
- Error conditions
- Serialization/deserialization
- Integration points

### What Not to Test (for v1)
- Internal helper functions
- Trivial getters/setters
- UI/CLI formatting

## For AI Subagents

### When You're Asked to Work on This Project

1. **Start by reading:**
   - `CONTEXT.md` - Full project context
   - `ROADMAP.md` - Current status and priorities
   - Relevant source files for your task

2. **Understand the scope:**
   - What specific task were you asked to do?
   - Is it in the current roadmap?
   - What files will you need to modify?

3. **Make focused changes:**
   - Do exactly what was asked
   - Don't refactor unrelated code
   - Don't add "nice to have" features
   - Keep changes minimal and testable

4. **Document what you did:**
   - Update ROADMAP.md if you completed a task
   - Add comments for non-obvious code
   - Note any issues you discovered

### Common Pitfalls to Avoid

```
❌ Rewriting large sections "for clarity"
❌ Adding features not in the roadmap
❌ Changing code style across the project
❌ Making assumptions about requirements
❌ Breaking existing functionality

✅ Making the specific change requested
✅ Asking for clarification if unsure
✅ Testing your changes
✅ Documenting discoveries
✅ Keeping the build green
```

### Handoff Template

When finishing a task, provide a summary:

```markdown
## Task Completed: [Task Name]

**What I did:**
- Implemented X in file Y
- Fixed bug in Z

**Files changed:**
- src/foo.rs (added function bar)
- src/baz.rs (fixed line 123)

**How to test:**
- Run `cargo test test_name`
- Or manually: [steps]

**Known issues:**
- None / [list any]

**Next steps:**
- [What should be done next]
```

## Project Structure Quick Reference

```
src/
├── lib.rs          # Public exports, core types
├── main.rs         # CLI entry point
├── runtime.rs      # Cortex struct - main runtime
├── session.rs      # Persistent session wrapper
├── config.rs       # All configuration types
│
├── agent/          # Agent management (v1)
│   ├── mod.rs      # Agent struct
│   └── manager.rs  # AgentManager
│
├── api/            # HTTP/WS API (v1)
│   ├── mod.rs      # Server setup
│   ├── routes.rs   # Route definitions
│   └── handlers.rs # Request handlers
│
├── memory/         # Memory subsystem
│   ├── mod.rs      # Memory interface
│   └── vector.rs   # Vector similarity store
│
├── inference/      # LLM inference
│   ├── mod.rs      # InferenceEngine trait
│   └── llama.rs    # llama.cpp backend
│
└── state/          # Persistence
    ├── mod.rs      # StateManager
    └── checkpoint.rs # Checkpoints, branches
```

## Getting Help

- **Confused about architecture?** → Read CONTEXT.md
- **Not sure what to work on?** → Check ROADMAP.md
- **Found a bug?** → Note it in ROADMAP.md under blockers
- **Have a question?** → Ask the human collaborator

## Definition of Done

A task is done when:
- [ ] Code compiles without warnings
- [ ] Tests pass
- [ ] Feature works as described
- [ ] ROADMAP.md updated
- [ ] No obvious bugs introduced
