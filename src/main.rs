//! Cortex CLI
//!
//! A command-line interface for the Cortex AI runtime.

use clap::{Parser, Subcommand};
use cortex::{Cortex, GenerationConfig, Message, Session};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "cortex")]
#[command(about = "The AI Runtime - Memory and state as primitives")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start an interactive chat session
    Chat {
        /// Path to the model file (GGUF format)
        #[arg(short, long)]
        model: PathBuf,

        /// Session ID for persistence
        #[arg(short, long)]
        session: Option<String>,

        /// System prompt
        #[arg(long)]
        system: Option<String>,

        /// Temperature (0.0 = deterministic, higher = more random)
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Maximum tokens to generate
        #[arg(long, default_value = "1024")]
        max_tokens: u32,
    },

    /// Generate a single completion
    Generate {
        /// Path to the model file (GGUF format)
        #[arg(short, long)]
        model: PathBuf,

        /// The prompt to complete
        prompt: String,

        /// Temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Maximum tokens
        #[arg(long, default_value = "256")]
        max_tokens: u32,
    },

    /// List all sessions
    Sessions,

    /// Delete a session
    DeleteSession {
        /// Session ID to delete
        session_id: String,
    },

    /// Show model info
    Info {
        /// Path to the model file
        #[arg(short, long)]
        model: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Chat {
            model,
            session,
            system,
            temperature,
            max_tokens,
        } => {
            run_chat(model, session, system, temperature, max_tokens)?;
        }

        Commands::Generate {
            model,
            prompt,
            temperature,
            max_tokens,
        } => {
            run_generate(model, prompt, temperature, max_tokens)?;
        }

        Commands::Sessions => {
            list_sessions()?;
        }

        Commands::DeleteSession { session_id } => {
            delete_session(&session_id)?;
        }

        Commands::Info { model } => {
            show_info(model)?;
        }
    }

    Ok(())
}

fn run_chat(
    model: PathBuf,
    session_id: Option<String>,
    system: Option<String>,
    temperature: f32,
    max_tokens: u32,
) -> anyhow::Result<()> {
    let config = GenerationConfig {
        temperature,
        max_tokens,
        ..Default::default()
    };

    if let Some(session_id) = session_id {
        // Use persistent session
        println!("Loading session '{}'...", session_id);
        // TODO: Load real model once llama-cpp is integrated
        let _ = model; // Suppress unused warning for now
        let mut session = Session::new(&session_id)?;

        if let Some(sys) = system {
            session.set_system(sys);
        }

        println!("Session loaded. Type 'quit' to exit, 'save' to save, 'clear' to clear.\n");
        run_chat_loop_session(&mut session, &config)?;
    } else {
        // One-off chat
        println!("Loading model...");
        let mut ctx = Cortex::load(&model)?;

        if let Some(sys) = system {
            ctx.chat(&[Message::system(sys)])?;
        }

        println!("Model loaded. Type 'quit' to exit.\n");
        run_chat_loop(&mut ctx, &config)?;
    }

    Ok(())
}

fn run_chat_loop(ctx: &mut Cortex, config: &GenerationConfig) -> anyhow::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("You: ");
        stdout.flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "quit" || input == "exit" {
            break;
        }

        print!("AI: ");
        stdout.flush()?;

        // Stream output
        let _response = ctx.chat_streaming(
            &[Message::user(input)],
            config,
            &mut |token| {
                print!("{}", token);
                stdout.flush().ok();
                true
            },
        )?;

        println!("\n");
    }

    Ok(())
}

fn run_chat_loop_session(session: &mut Session, _config: &GenerationConfig) -> anyhow::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("You: ");
        stdout.flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        match input {
            "quit" | "exit" => break,
            "save" => {
                session.save()?;
                println!("Session saved.");
                continue;
            }
            "clear" => {
                session.clear()?;
                println!("Session cleared.");
                continue;
            }
            _ => {}
        }

        print!("AI: ");
        stdout.flush()?;

        // Stream output
        let _response = session.chat_streaming(input, &mut |token| {
            print!("{}", token);
            stdout.flush().ok();
            true
        })?;

        println!("\n");
    }

    Ok(())
}

fn run_generate(
    model: PathBuf,
    prompt: String,
    temperature: f32,
    max_tokens: u32,
) -> anyhow::Result<()> {
    println!("Loading model...");
    let mut ctx = Cortex::load(&model)?;

    let config = GenerationConfig {
        temperature,
        max_tokens,
        ..Default::default()
    };

    println!("Generating...\n");

    let mut stdout = io::stdout();
    let _response = ctx.generate_streaming(&prompt, &config, &mut |token| {
        print!("{}", token);
        stdout.flush().ok();
        true
    })?;

    println!("\n");
    Ok(())
}

fn list_sessions() -> anyhow::Result<()> {
    let sessions = cortex::session::list_sessions()?;

    if sessions.is_empty() {
        println!("No sessions found.");
    } else {
        println!("Sessions:");
        for session in sessions {
            println!("  - {}", session);
        }
    }

    Ok(())
}

fn delete_session(session_id: &str) -> anyhow::Result<()> {
    cortex::session::delete_session(session_id)?;
    println!("Session '{}' deleted.", session_id);
    Ok(())
}

fn show_info(model: PathBuf) -> anyhow::Result<()> {
    println!("Loading model...");
    let ctx = Cortex::load(&model)?;

    println!("\nModel Information:");
    println!("  Context size: {} tokens", ctx.context_size());
    println!("  Embedding dim: {}", ctx.embedding_dim());
    println!("  Memory entries: {}", ctx.memory.len());

    Ok(())
}
