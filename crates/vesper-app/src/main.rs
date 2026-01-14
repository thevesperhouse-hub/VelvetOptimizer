//! VesperAI Tauri V2 Application
//! 
//! Desktop UI for interactive LLM training

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod commands;

fn main() {
    tauri::Builder::default()
        .manage(commands::AppState::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            commands::check_cuda,
            commands::start_training,
            commands::get_training_status,
            commands::stop_training,
            commands::load_dataset,
            commands::get_model_configs,
            commands::auto_scale,
            commands::search_hf_datasets,
            commands::load_hf_dataset,
            commands::detect_dataset_format,
            commands::start_benchmark,
            commands::get_benchmark_results,
            commands::save_model,
            commands::export_onnx,
            commands::generate_text,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
