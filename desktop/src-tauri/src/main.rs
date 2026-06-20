// Prevents an extra console window on Windows in release. Harmless on Linux.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::Duration;

use tauri::{Manager, WebviewUrl, WebviewWindowBuilder, WindowEvent};

/// Self-contained loading screen shown while the Docker stack boots.
/// Dark bg #0a0e1a, monospace, green #1D9E75 text, animated dots.
const LOADING_HTML: &str = r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  html,body{height:100%;margin:0}
  body{background:#0a0e1a;color:#1D9E75;
       font-family:"JetBrains Mono","Fira Code",ui-monospace,Menlo,Consolas,monospace;
       display:flex;align-items:center;justify-content:center}
  .t{font-size:18px;letter-spacing:1px}
  .dots span{animation:blink 1.2s infinite}
  .dots span:nth-child(2){animation-delay:.2s}
  .dots span:nth-child(3){animation-delay:.4s}
  @keyframes blink{0%,100%{opacity:.2}50%{opacity:1}}
</style></head>
<body><div class="t">// starting<span class="dots"><span>.</span><span>.</span><span>.</span></span></div></body></html>"#;

/// Directory that holds docker-compose.yml. Overridable via FILE_ASSISTANT_DIR.
fn project_root() -> PathBuf {
    std::env::var("FILE_ASSISTANT_DIR")
        .unwrap_or_else(|_| "/home/jf/Desktop/file-assistant".to_string())
        .into()
}

/// Run `docker compose <args>` from the project root, waiting for completion.
fn docker_compose(args: &[&str]) {
    let status = Command::new("docker")
        .arg("compose")
        .args(args)
        .current_dir(project_root())
        .status();
    match status {
        Ok(s) => println!("// docker compose {:?} -> {}", args, s),
        Err(e) => eprintln!("// docker compose {:?} failed to launch: {}", args, e),
    }
}

/// True once GET /health returns {"status":"ok"}.
fn health_ok() -> bool {
    match ureq::get("http://localhost:8000/health")
        .timeout(Duration::from_secs(2))
        .call()
    {
        Ok(resp) => resp
            .into_string()
            .map(|s| s.contains("\"status\":\"ok\""))
            .unwrap_or(false),
        Err(_) => false,
    }
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // Show the loading screen from a temp file via file:// (self-contained).
            let loading_path = std::env::temp_dir().join("file-assistant-loading.html");
            std::fs::write(&loading_path, LOADING_HTML).ok();
            let loading_url = format!("file://{}", loading_path.display());

            WebviewWindowBuilder::new(
                app,
                "loading",
                WebviewUrl::External(loading_url.parse().expect("loading url")),
            )
            .title("file-assistant")
            .inner_size(440.0, 260.0)
            .resizable(false)
            .decorations(false)
            .center()
            .build()?;

            // Boot the stack and wait for health off the UI thread.
            let handle = app.handle().clone();
            thread::spawn(move || {
                println!("// starting docker stack...");
                docker_compose(&["up", "-d"]);

                // Poll /health every second until the API reports ok.
                while !health_ok() {
                    thread::sleep(Duration::from_secs(1));
                }
                println!("// health ok — opening main window");

                // Reveal the main window (loads http://localhost:8000) and drop loading.
                if let Some(main) = handle.get_webview_window("main") {
                    let _ = main.show();
                    let _ = main.set_focus();
                }
                if let Some(loading) = handle.get_webview_window("loading") {
                    let _ = loading.close();
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            // Clean shutdown: when the main window is closed, bring the Docker
            // stack down and wait for it before the process exits.
            if window.label() == "main" {
                if let WindowEvent::CloseRequested { .. } = event {
                    println!("// stopping docker stack...");
                    docker_compose(&["down"]);
                    println!("// stack stopped");
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
