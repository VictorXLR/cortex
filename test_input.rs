use std::io::{self, BufRead, Write};

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    
    loop {
        print!("Test: ");
        stdout.flush()?;
        
        let mut input = String::new();
        match stdin.lock().read_line(&mut input) {
            Ok(0) => {
                println!("EOF received");
                break;
            }
            Ok(_) => {
                let input = input.trim();
                println!("Got: {}", input);
                if input == "quit" {
                    break;
                }
            }
            Err(e) => {
                println!("Error: {}", e);
                break;
            }
        }
    }
    
    Ok(())
}