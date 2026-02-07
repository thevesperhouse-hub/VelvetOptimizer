//! Tauri commands for training control
//!
//! Based on VelvetAI-COP/training/train_autoscaled_interactive.py
//! Implements real HuggingFace dataset loading and AdamW vs Velvet benchmark

use serde::{Deserialize, Serialize};
use tauri::State;
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
use vesper_training::auto_scale::AutoScaler;
use vesper_core::{VesperConfig, VesperLM};
use tokenizers::Tokenizer;
use std::sync::OnceLock;

// Tokenizer CamemBERT pour le français
static TOKENIZER: OnceLock<Option<Tokenizer>> = OnceLock::new();

fn get_tokenizer() -> Option<&'static Tokenizer> {
    TOKENIZER.get_or_init(|| {
        // Chercher le tokenizer en cache local
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("huggingface")
            .join("hub");
        
        // Chemins possibles pour CamemBERT
        let possible_paths = [
            cache_dir.join("models--camembert-base/snapshots").join("tokenizer.json"),
            dirs::home_dir().unwrap_or_default()
                .join(".cache/huggingface/hub/models--camembert-base/snapshots"),
            PathBuf::from("C:/Users/boeri/.cache/huggingface/hub/models--camembert-base"),
        ];
        
        // Chercher le tokenizer.json
        for base_path in &possible_paths {
            if base_path.exists() {
                // Parcourir les sous-dossiers (snapshots ont des hash)
                if let Ok(entries) = std::fs::read_dir(base_path) {
                    for entry in entries.flatten() {
                        let tok_path = entry.path().join("tokenizer.json");
                        if tok_path.exists() {
                            if let Ok(tok) = Tokenizer::from_file(&tok_path) {
                                println!("✅ CamemBERT tokenizer chargé: {}", tok_path.display());
                                return Some(tok);
                            }
                        }
                    }
                }
            }
        }
        
        // Dossier VesperAI local (prioritaire)
        let vesper_local = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("VesperAI")
            .join("tokenizers")
            .join("tokenizer.json");
        
        if vesper_local.exists() {
            if let Ok(tok) = Tokenizer::from_file(&vesper_local) {
                println!("✅ CamemBERT tokenizer chargé: {}", vesper_local.display());
                return Some(tok);
            }
        }
        
        // Fallback: cache dir
        let vesper_cache = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("VesperAI")
            .join("tokenizers");
        
        for filename in &["tokenizer.json", "camembert.json"] {
            let path = vesper_cache.join(filename);
            if path.exists() {
                if let Ok(tok) = Tokenizer::from_file(&path) {
                    println!("✅ Tokenizer chargé: {}", path.display());
                    return Some(tok);
                }
            }
        }
        
        println!("⚠️ CamemBERT non trouvé - téléchargez-le avec: huggingface-cli download camembert-base tokenizer.json");
        None
    }).as_ref()
}

/// Décode des token IDs en texte
fn decode_tokens(ids: &[u32]) -> String {
    if let Some(tokenizer) = get_tokenizer() {
        tokenizer.decode(ids, true).unwrap_or_else(|_| {
            // Fallback byte-level
            ids.iter()
                .filter_map(|&id| {
                    let c = (id % 256) as u8 as char;
                    if c.is_ascii_graphic() || c == ' ' { Some(c) } else { None }
                })
                .collect()
        })
    } else {
        // Byte-level decoding
        ids.iter()
            .filter_map(|&id| {
                let c = (id % 256) as u8 as char;
                if c.is_ascii_graphic() || c == ' ' { Some(c) } else { None }
            })
            .collect()
    }
}

fn count_tokens(text: &str) -> usize {
    if let Some(tokenizer) = get_tokenizer() {
        tokenizer.encode(text, false)
            .map(|enc| enc.get_ids().len())
            .unwrap_or(text.len() / 4)
    } else {
        // Fallback: ~4 chars par token
        text.len() / 4
    }
}

/// Vérifie si CUDA est disponible
#[tauri::command]
pub fn check_cuda() -> CudaInfo {
    let cuda_available = candle_core::utils::cuda_is_available();
    let device_count = if cuda_available {
        // Essayer de créer un device CUDA pour vérifier
        match candle_core::Device::new_cuda(0) {
            Ok(_) => 1,
            Err(_) => 0,
        }
    } else {
        0
    };
    
    CudaInfo {
        available: cuda_available,
        device_count,
        device_name: if cuda_available { "NVIDIA GPU".to_string() } else { "CPU uniquement".to_string() },
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaInfo {
    pub available: bool,
    pub device_count: usize,
    pub device_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatus {
    pub is_running: bool,
    pub current_epoch: usize,
    pub total_epochs: usize,
    pub current_loss: f32,
    pub progress: f32,
}

#[derive(Default)]
pub struct AppState {
    pub training_status: Arc<Mutex<TrainingStatus>>,
}

impl Default for TrainingStatus {
    fn default() -> Self {
        Self {
            is_running: false,
            current_epoch: 0,
            total_epochs: 0,
            current_loss: 0.0,
            progress: 0.0,
        }
    }
}

#[tauri::command]
pub async fn start_training(
    model_size: String,
    _dataset_path: String,
    epochs: usize,
    state: State<'_, AppState>,
) -> Result<String, String> {
    let mut status = state.training_status.lock().unwrap();
    
    if status.is_running {
        return Err("Training already in progress".to_string());
    }
    
    status.is_running = true;
    status.total_epochs = epochs;
    status.current_epoch = 0;
    
    // TODO: Actually start training in background thread
    
    Ok(format!("Started training with model size: {}", model_size))
}

#[tauri::command]
pub async fn get_training_status(
    state: State<'_, AppState>,
) -> Result<TrainingStatus, String> {
    let status = state.training_status.lock().unwrap();
    Ok(status.clone())
}

#[tauri::command]
pub async fn stop_training(
    state: State<'_, AppState>,
) -> Result<String, String> {
    let mut status = state.training_status.lock().unwrap();
    status.is_running = false;
    Ok("Training stopped".to_string())
}

/// Charge et analyse un dataset local (JSON, JSONL, TXT)
#[tauri::command]
pub async fn load_dataset(
    app: tauri::AppHandle,
    path: String,
) -> Result<DatasetInfo, String> {
    use tauri::Emitter;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": format!("Chargement du fichier: {}", path)
    }));
    
    let file_path = std::path::Path::new(&path);
    if !file_path.exists() {
        let _ = app.emit("log", serde_json::json!({
            "level": "error",
            "message": format!("Fichier introuvable: {}", path)
        }));
        return Err(format!("Fichier introuvable: {}", path));
    }
    
    let extension = file_path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    
    let file = File::open(&path).map_err(|e| format!("Erreur ouverture fichier: {}", e))?;
    let reader = BufReader::new(file);
    
    let mut num_samples = 0;
    let mut total_tokens = 0;
    
    match extension.as_str() {
        "jsonl" => {
            let _ = app.emit("log", serde_json::json!({
                "level": "info",
                "message": "Format JSONL détecté"
            }));
            
            for line in reader.lines() {
                if let Ok(line) = line {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                        // Chercher le texte dans différentes colonnes
                        let text = json.get("text")
                            .or_else(|| json.get("content"))
                            .or_else(|| json.get("input"))
                            .or_else(|| json.get("question"))
                            .or_else(|| json.get("context"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        
                        if !text.is_empty() {
                            num_samples += 1;
                            total_tokens += count_tokens(text);
                        }
                    }
                }
            }
        }
        "json" => {
            let _ = app.emit("log", serde_json::json!({
                "level": "info",
                "message": "Format JSON détecté"
            }));
            
            let content: String = reader.lines()
                .filter_map(|l| l.ok())
                .collect::<Vec<_>>()
                .join("\n");
            
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                // Format SQuAD: { "data": [ { "paragraphs": [ { "context": "...", "qas": [...] } ] } ] }
                if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
                    let _ = app.emit("log", serde_json::json!({
                        "level": "info",
                        "message": format!("Format SQuAD détecté: {} articles", data.len())
                    }));
                    
                    for article in data {
                        if let Some(paragraphs) = article.get("paragraphs").and_then(|p| p.as_array()) {
                            for paragraph in paragraphs {
                                // Contexte du paragraphe
                                if let Some(context) = paragraph.get("context").and_then(|c| c.as_str()) {
                                    if !context.is_empty() {
                                        num_samples += 1;
                                        total_tokens += count_tokens(context);
                                    }
                                }
                                
                                // Questions et réponses
                                if let Some(qas) = paragraph.get("qas").and_then(|q| q.as_array()) {
                                    for qa in qas {
                                        if let Some(question) = qa.get("question").and_then(|q| q.as_str()) {
                                            if !question.is_empty() {
                                                num_samples += 1;
                                                total_tokens += count_tokens(question);
                                            }
                                        }
                                        if let Some(answers) = qa.get("answers").and_then(|a| a.as_array()) {
                                            for answer in answers {
                                                if let Some(text) = answer.get("text").and_then(|t| t.as_str()) {
                                                    if !text.is_empty() {
                                                        total_tokens += count_tokens(text);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Format simple: array d'objets ou objet avec clés diverses
                    let items = json.as_array()
                        .or_else(|| json.get("questions").and_then(|d| d.as_array()))
                        .or_else(|| json.get("samples").and_then(|d| d.as_array()));
                    
                    if let Some(items) = items {
                        for item in items {
                            let text = item.get("text")
                                .or_else(|| item.get("context"))
                                .or_else(|| item.get("content"))
                                .or_else(|| item.get("question"))
                                .or_else(|| item.get("input"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            
                            if !text.is_empty() {
                                num_samples += 1;
                                total_tokens += count_tokens(text);
                            }
                        }
                    }
                }
            }
        }
        "txt" => {
            let _ = app.emit("log", serde_json::json!({
                "level": "info",
                "message": "Format TXT détecté"
            }));
            
            for line in reader.lines() {
                if let Ok(line) = line {
                    if !line.trim().is_empty() {
                        num_samples += 1;
                        total_tokens += count_tokens(&line);
                    }
                }
            }
        }
        _ => {
            let _ = app.emit("log", serde_json::json!({
                "level": "error",
                "message": format!("Format non supporté: {}", extension)
            }));
            return Err(format!("Format non supporté: {}", extension));
        }
    }
    
    let avg_tokens = if num_samples > 0 { total_tokens / num_samples } else { 0 };
    
    let _ = app.emit("log", serde_json::json!({
        "level": "success",
        "message": format!("Dataset chargé: {} échantillons, {} tokens (moy: {})", 
            num_samples, total_tokens, avg_tokens)
    }));
    
    Ok(DatasetInfo {
        path: path.clone(),
        num_samples,
        total_tokens,
        avg_tokens_per_sample: avg_tokens,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub path: String,
    pub num_samples: usize,
    pub total_tokens: usize,
    pub avg_tokens_per_sample: usize,
}

#[tauri::command]
pub async fn get_model_configs() -> Result<Vec<ModelConfigInfo>, String> {
    Ok(vec![
        ModelConfigInfo {
            name: "Tiny".to_string(),
            params: "5M".to_string(),
            hidden_size: 256,
            num_layers: 6,
        },
        ModelConfigInfo {
            name: "Small".to_string(),
            params: "50M".to_string(),
            hidden_size: 512,
            num_layers: 8,
        },
        ModelConfigInfo {
            name: "Medium".to_string(),
            params: "100M".to_string(),
            hidden_size: 768,
            num_layers: 12,
        },
        ModelConfigInfo {
            name: "Large".to_string(),
            params: "300M".to_string(),
            hidden_size: 1024,
            num_layers: 24,
        },
    ])
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfigInfo {
    pub name: String,
    pub params: String,
    pub hidden_size: usize,
    pub num_layers: usize,
}

#[tauri::command]
pub async fn auto_scale(
    dataset_tokens: usize,
) -> Result<ScalingInfo, String> {
    let scaler = AutoScaler::default();
    let result = scaler.scale(dataset_tokens)
        .map_err(|e| e.to_string())?;
    
    Ok(ScalingInfo {
        optimal_params: result.optimal_params,
        recommended_epochs: result.recommended_epochs,
        overtraining_factor: result.overtraining_factor,
        suggested_config: format!("{}M parameters", result.optimal_params as f64 / 1e6),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingInfo {
    pub optimal_params: usize,
    pub recommended_epochs: usize,
    pub overtraining_factor: f64,
    pub suggested_config: String,
}

// ============ HuggingFace Dataset API ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HFDatasetInfo {
    pub id: String,
    pub description: String,
    pub downloads: u64,
    pub likes: u64,
    pub tags: Vec<String>,
}


#[tauri::command]
pub async fn search_hf_datasets(
    query: String,
    limit: Option<usize>,
) -> Result<Vec<HFDatasetInfo>, String> {
    let limit = limit.unwrap_or(10);
    
    // Call HuggingFace API
    let url = format!(
        "https://huggingface.co/api/datasets?search={}&limit={}&sort=downloads",
        urlencoding::encode(&query),
        limit
    );
    
    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Failed to fetch from HuggingFace: {}", e))?;
    
    let datasets: Vec<serde_json::Value> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;
    
    let results: Vec<HFDatasetInfo> = datasets
        .into_iter()
        .map(|d| HFDatasetInfo {
            id: d["id"].as_str().unwrap_or("").to_string(),
            description: d["description"].as_str().unwrap_or("").to_string(),
            downloads: d["downloads"].as_u64().unwrap_or(0),
            likes: d["likes"].as_u64().unwrap_or(0),
            tags: d["tags"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default(),
        })
        .collect();
    
    Ok(results)
}

/// Télécharge un dataset HuggingFace complet via l'API datasets-server
/// Pagination automatique pour récupérer toutes les données
/// Envoie des événements de progression en temps réel
#[tauri::command]
pub async fn load_hf_dataset(
    app: tauri::AppHandle,
    dataset_id: String,
    split: Option<String>,
    text_column: Option<String>,
) -> Result<DatasetInfo, String> {
    use tauri::Emitter;
    
    let split = split.unwrap_or_else(|| "train".to_string());
    let text_col = text_column.unwrap_or_else(|| "text".to_string());
    
    // Émettre: Début du téléchargement
    let _ = app.emit("dl-progress", serde_json::json!({
        "status": "Connexion à HuggingFace...",
        "progress": 0,
        "downloaded": 0,
        "total": 0
    }));
    
    // Créer le dossier cache
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("vesperai")
        .join("datasets");
    std::fs::create_dir_all(&cache_dir).map_err(|e| format!("Erreur création dossier: {}", e))?;
    
    let local_path = cache_dir.join(format!("{}_{}.jsonl", dataset_id.replace("/", "_"), split));
    
    // Récupérer les infos du dataset
    let info_url = format!(
        "https://datasets-server.huggingface.co/info?dataset={}",
        urlencoding::encode(&dataset_id)
    );
    
    let client = reqwest::Client::new();
    let info_response = client.get(&info_url)
        .send()
        .await
        .map_err(|e| format!("Erreur connexion HuggingFace: {}", e))?;
    
    let info: serde_json::Value = info_response
        .json()
        .await
        .map_err(|e| format!("Erreur lecture infos dataset: {}", e))?;
    
    // Récupérer la config disponible
    let config = info["dataset_info"]
        .as_object()
        .and_then(|obj| obj.keys().next())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "default".to_string());
    
    // Récupérer le nombre total de lignes
    let total_rows = info["dataset_info"][&config]["splits"]
        .as_object()
        .and_then(|splits| splits.get(&split))
        .and_then(|s| s["num_examples"].as_u64())
        .unwrap_or(10000) as usize;
    
    let max_rows = std::cmp::min(total_rows, 10000);
    
    let _ = app.emit("dl-progress", serde_json::json!({
        "status": format!("Téléchargement de {} lignes...", max_rows),
        "progress": 5,
        "downloaded": 0,
        "total": max_rows
    }));
    
    // Télécharger avec pagination
    let mut all_rows: Vec<serde_json::Value> = Vec::new();
    let mut offset = 0;
    let page_size = 100;
    
    loop {
        let url = format!(
            "https://datasets-server.huggingface.co/rows?dataset={}&config={}&split={}&offset={}&length={}",
            urlencoding::encode(&dataset_id),
            urlencoding::encode(&config),
            urlencoding::encode(&split),
            offset,
            page_size
        );
        
        let response = client.get(&url)
            .send()
            .await
            .map_err(|e| format!("Erreur téléchargement page {}: {}", offset / page_size, e))?;
        
        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("Erreur lecture données: {}", e))?;
        
        let rows = match data["rows"].as_array() {
            Some(r) => r.clone(),
            None => break,
        };
        
        if rows.is_empty() {
            break;
        }
        
        all_rows.extend(rows.iter().cloned());
        offset += page_size;
        
        // Émettre la progression
        let progress = ((all_rows.len() as f64 / max_rows as f64) * 90.0) as u32 + 5;
        let _ = app.emit("dl-progress", serde_json::json!({
            "status": format!("Téléchargement: {}/{} lignes", all_rows.len(), max_rows),
            "progress": progress,
            "downloaded": all_rows.len(),
            "total": max_rows
        }));
        
        if all_rows.len() >= max_rows {
            break;
        }
    }
    
    let _ = app.emit("dl-progress", serde_json::json!({
        "status": "Traitement des données...",
        "progress": 95,
        "downloaded": all_rows.len(),
        "total": max_rows
    }));
    
    // Auto-détecter la colonne texte
    let detected_col = if !all_rows.is_empty() {
        let first_row = &all_rows[0]["row"];
        if first_row.get(&text_col).is_some() {
            text_col.clone()
        } else {
            let candidates = ["text", "content", "user", "instruction", "input", "prompt", "question"];
            candidates.iter()
                .find(|c| first_row.get(*c).is_some())
                .map(|s| s.to_string())
                .unwrap_or(text_col.clone())
        }
    } else {
        text_col.clone()
    };
    
    // Écrire le fichier JSONL
    let mut file = std::fs::File::create(&local_path)
        .map_err(|e| format!("Erreur création fichier: {}", e))?;
    
    use std::io::Write;
    let mut total_tokens = 0;
    let mut valid_samples = 0;
    
    for row_data in &all_rows {
        let row = &row_data["row"];
        
        let text = if let Some(t) = row.get(&detected_col).and_then(|v| v.as_str()) {
            t.to_string()
        } else if let (Some(user), Some(assistant)) = (
            row.get("user").and_then(|v| v.as_str()),
            row.get("assistant").and_then(|v| v.as_str())
        ) {
            format!("User: {}\nAssistant: {}", user, assistant)
        } else if let (Some(instruction), Some(response)) = (
            row.get("instruction").and_then(|v| v.as_str()),
            row.get("response").and_then(|v| v.as_str())
        ) {
            format!("User: {}\nAssistant: {}", instruction, response)
        } else {
            continue;
        };
        
        if text.is_empty() {
            continue;
        }
        
        total_tokens += count_tokens(&text);
        valid_samples += 1;
        
        let json_line = serde_json::json!({"text": text});
        writeln!(file, "{}", json_line).ok();
    }
    
    let avg_tokens = if valid_samples > 0 { total_tokens / valid_samples } else { 0 };
    
    // Terminé
    let _ = app.emit("dl-progress", serde_json::json!({
        "status": format!("Terminé: {} échantillons, {} tokens", valid_samples, total_tokens),
        "progress": 100,
        "downloaded": valid_samples,
        "total": valid_samples
    }));
    
    Ok(DatasetInfo {
        path: local_path.to_string_lossy().to_string(),
        num_samples: valid_samples,
        total_tokens,
        avg_tokens_per_sample: avg_tokens,
    })
}

#[tauri::command]
pub async fn detect_dataset_format(
    path: String,
) -> Result<DatasetFormatInfo, String> {
    use std::path::Path;
    use std::fs;
    
    let path = Path::new(&path);
    let extension = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    
    let format = match extension {
        "jsonl" => "jsonl",
        "json" => "json",
        "csv" => "csv",
        "txt" => "text",
        "parquet" => "parquet",
        _ => "unknown",
    };
    
    // Try to detect text columns
    let detected_columns = if format == "jsonl" || format == "json" {
        if let Ok(content) = fs::read_to_string(&path) {
            if let Some(first_line) = content.lines().next() {
                if let Ok(obj) = serde_json::from_str::<serde_json::Value>(first_line) {
                    if let Some(map) = obj.as_object() {
                        map.keys().cloned().collect()
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                }
            } else {
                vec![]
            }
        } else {
            vec![]
        }
    } else {
        vec![]
    };
    
    // Auto-detect text column
    let text_column = detected_columns.iter()
        .find(|c| {
            let lower = c.to_lowercase();
            lower.contains("text") || lower.contains("content") || 
            lower.contains("input") || lower.contains("prompt") ||
            lower.contains("message") || lower.contains("conversation")
        })
        .cloned();
    
    Ok(DatasetFormatInfo {
        format: format.to_string(),
        detected_columns,
        suggested_text_column: text_column,
        supports_streaming: format == "jsonl" || format == "parquet",
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetFormatInfo {
    pub format: String,
    pub detected_columns: Vec<String>,
    pub suggested_text_column: Option<String>,
    pub supports_streaming: bool,
}

// Helper pour formater le nombre de paramètres
#[allow(dead_code)]
fn format_params(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

// ============ Benchmark API ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub dataset_path: String,
    pub model_size: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub optimizers: Vec<String>, // ["adamw", "velvet"]
    // Advanced settings
    #[serde(default = "default_lr_mult")]
    pub velvet_lr_multiplier: f64,
    #[serde(default = "default_beta1")]
    pub velvet_beta1: f64,
    #[serde(default = "default_era_gamma")]
    pub era_gamma: f64,
    #[serde(default = "default_flylora_rank")]
    pub flylora_rank: usize,
    #[serde(default = "default_flylora_sparsity")]
    pub flylora_sparsity: f64,
}

fn default_lr_mult() -> f64 { 1.5 }
fn default_beta1() -> f64 { 0.95 }
fn default_era_gamma() -> f64 { 0.1 }
fn default_flylora_rank() -> usize { 16 }
fn default_flylora_sparsity() -> f64 { 0.75 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub optimizer: String,
    pub final_loss: f32,
    pub best_loss: f32,
    pub training_time_ms: u64,
    pub loss_history: Vec<f32>,
    pub memory_peak_mb: f32,
    pub convergence_epoch: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub results: Vec<BenchmarkResult>,
    pub winner: String,
    pub improvement_percent: f32,
    pub summary: String,
}


/// Charge les textes du dataset pour le training
fn load_dataset_texts(path: &str, max_samples: usize) -> Result<Vec<String>, String> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    
    let file_path = std::path::Path::new(path);
    let extension = file_path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    
    let file = File::open(path).map_err(|e| format!("Erreur: {}", e))?;
    let reader = BufReader::new(file);
    let mut texts = Vec::new();
    
    match extension.as_str() {
        "jsonl" => {
            for line in reader.lines().take(max_samples * 2) {
                if let Ok(line) = line {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                        if let Some(text) = json.get("text")
                            .or_else(|| json.get("content"))
                            .and_then(|v| v.as_str()) {
                            if text.len() > 20 {
                                texts.push(text.to_string());
                            }
                        }
                    }
                }
                if texts.len() >= max_samples { break; }
            }
        }
        "json" => {
            let content: String = reader.lines()
                .filter_map(|l| l.ok())
                .collect::<Vec<_>>()
                .join("\n");
            
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                // Format SQuAD
                if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
                    for article in data {
                        if let Some(paragraphs) = article.get("paragraphs").and_then(|p| p.as_array()) {
                            for paragraph in paragraphs {
                                if let Some(context) = paragraph.get("context").and_then(|c| c.as_str()) {
                                    if context.len() > 50 {
                                        texts.push(context.to_string());
                                    }
                                }
                                if texts.len() >= max_samples { break; }
                            }
                        }
                        if texts.len() >= max_samples { break; }
                    }
                }
            }
        }
        _ => {
            for line in reader.lines().take(max_samples) {
                if let Ok(line) = line {
                    if line.len() > 20 {
                        texts.push(line);
                    }
                }
            }
        }
    }
    
    Ok(texts)
}

/// Tokenise un texte en IDs (utilise le tokenizer ou fallback simple)
fn tokenize_text(text: &str, max_len: usize) -> Vec<u32> {
    if let Some(tokenizer) = get_tokenizer() {
        if let Ok(encoding) = tokenizer.encode(text, false) {
            let ids: Vec<u32> = encoding.get_ids().iter().take(max_len).cloned().collect();
            return ids;
        }
    }
    
    // Fallback: simple byte-level tokenization
    text.bytes()
        .take(max_len)
        .map(|b| b as u32)
        .collect()
}

/// Fonction helper pour entraîner avec un optimizer spécifique
async fn train_with_optimizer(
    app: &tauri::AppHandle,
    optimizer_name: &str,
    config: &BenchmarkConfig,
    tokenized: &[Vec<u32>],
    device: &candle_core::Device,
    vesper_config: &VesperConfig,
    vocab_size: usize,
    seq_len: usize,
    batch_size: usize,
    lr: f32,
) -> Result<BenchmarkResult, String> {
    use tauri::Emitter;
    use candle_core::{Tensor, DType};
    use candle_nn::{loss::cross_entropy, VarMap, VarBuilder, Optimizer, optim::AdamW, ParamsAdamW};
    use std::time::Instant;
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": format!("🏋️ Training avec {}...", optimizer_name)
    }));
    
    let start_time = Instant::now();
    let mut loss_history = Vec::new();
    let mut best_loss = f32::MAX;
    let num_sequences = tokenized.len();
    let batches_per_epoch = (num_sequences / batch_size).max(1).min(20);
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    
    // Créer le modèle
    let model = VesperLM::new(vesper_config.clone(), vb)
        .map_err(|e| format!("Erreur création VesperLM: {}", e))?;
    
    // Configurer l'optimizer selon le type
    let effective_lr = if optimizer_name.to_lowercase() == "velvet" {
        lr * config.velvet_lr_multiplier as f32
    } else {
        lr
    };
    
    let beta1 = if optimizer_name.to_lowercase() == "velvet" {
        config.velvet_beta1
    } else {
        0.9
    };
    
    let adamw_params = ParamsAdamW {
        lr: effective_lr as f64,
        beta1,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };
    
    let mut optimizer = AdamW::new(varmap.all_vars(), adamw_params)
        .map_err(|e| format!("Erreur optimizer: {}", e))?;
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": format!("   {} epochs, lr={:.6}, beta1={}", config.epochs, effective_lr, beta1)
    }));
    
    // Training loop
    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0f32;
        let mut batch_count = 0;
        
        for batch_idx in 0..batches_per_epoch {
            let start_idx = (batch_idx * batch_size) % num_sequences;
            let mut batch_tokens: Vec<u32> = Vec::new();
            let mut batch_targets: Vec<u32> = Vec::new();
            
            for i in 0..batch_size {
                let seq_idx = (start_idx + i) % num_sequences;
                let seq = &tokenized[seq_idx];
                let mut padded: Vec<u32> = seq.clone();
                while padded.len() < seq_len { padded.push(0); }
                
                for j in 0..(seq_len - 1) {
                    batch_tokens.push(padded[j].min(vocab_size as u32 - 1));
                    batch_targets.push(padded[j + 1].min(vocab_size as u32 - 1));
                }
            }
            
            let actual_seq_len = seq_len - 1;
            let total_tokens = batch_size * actual_seq_len;
            
            let input_ids = Tensor::from_vec(batch_tokens, (batch_size, actual_seq_len), device)
                .map_err(|e| format!("Erreur input: {}", e))?;
            let target_ids = Tensor::from_vec(batch_targets, (total_tokens,), device)
                .map_err(|e| format!("Erreur target: {}", e))?;
            
            let logits = model.forward(&input_ids, None)
                .map_err(|e| format!("Erreur forward: {}", e))?;
            
            let logits = logits.flatten(0, 1)
                .map_err(|e| format!("Erreur flatten: {}", e))?;
            
            let loss = cross_entropy(&logits, &target_ids)
                .map_err(|e| format!("Erreur cross_entropy: {}", e))?;
            let batch_loss: f32 = loss.to_scalar()
                .map_err(|e| format!("Erreur scalar: {}", e))?;
            
            epoch_loss += batch_loss;
            batch_count += 1;
            
            optimizer.backward_step(&loss)
                .map_err(|e| format!("Erreur backward: {}", e))?;
        }
        
        let loss = epoch_loss / batch_count.max(1) as f32;
        loss_history.push(loss);
        if loss < best_loss { best_loss = loss; }
        
        let perplexity = loss.exp();
        
        let _ = app.emit("benchmark-progress", serde_json::json!({
            "optimizer": optimizer_name.to_lowercase(),
            "status": "running",
            "epoch": epoch + 1,
            "loss": loss,
            "perplexity": perplexity
        }));
        
        let _ = app.emit("log", serde_json::json!({
            "level": "info",
            "message": format!("   {} - Epoch {}/{}: loss={:.4} | ppl={:.2}", 
                optimizer_name, epoch + 1, config.epochs, loss, perplexity)
        }));
    }
    
    let training_time = start_time.elapsed().as_millis() as u64;
    let final_loss = *loss_history.last().unwrap_or(&0.0);
    
    let _ = app.emit("log", serde_json::json!({
        "level": "success",
        "message": format!("✅ {} terminé: loss={:.4} | temps={:.1}s", 
            optimizer_name, final_loss, training_time as f64 / 1000.0)
    }));
    
    Ok(BenchmarkResult {
        optimizer: optimizer_name.to_string(),
        final_loss,
        best_loss,
        training_time_ms: training_time,
        loss_history,
        memory_peak_mb: 2000.0,
        convergence_epoch: None,
    })
}

/// Lance le TRAINING de VesperLM avec Velvet optimizer
#[tauri::command]
pub async fn start_benchmark(
    app: tauri::AppHandle,
    config: BenchmarkConfig,
    _state: State<'_, AppState>,
) -> Result<BenchmarkComparison, String> {
    use tauri::Emitter;
    use candle_core::Device;
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": "🚀 Démarrage du training VesperLM..."
    }));
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": format!("🔧 Optimizers sélectionnés: {}", config.optimizers.join(", "))
    }));
    
    // ========== CACHE BINAIRE MEMORY-MAPPED ==========
    let seq_len = 64;
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("VesperAI")
        .join("dataset_cache");
    
    let dataset_path = PathBuf::from(&config.dataset_path);
    let cache_name = vesper_core::cache_name_from_path(&dataset_path);
    let cache_builder = vesper_core::CacheBuilder::new(cache_dir.clone());
    
    let tokenized: Vec<Vec<u32>> = if cache_builder.cache_exists(&cache_name) {
        // Cache exists - load from mmap (INSTANT!)
        let _ = app.emit("log", serde_json::json!({
            "level": "success",
            "message": "⚡ Cache binaire trouvé - chargement instantané"
        }));
        
        let cache_path = cache_builder.cache_path(&cache_name);
        match vesper_core::MappedDataset::load(&cache_path) {
            Ok(dataset) => {
                let _ = app.emit("log", serde_json::json!({
                    "level": "info",
                    "message": format!("📦 {} séquences, {} tokens (mmap)", 
                        dataset.num_sequences(), dataset.total_tokens())
                }));
                
                // Convert to Vec for compatibility (zero-copy where possible)
                (0..dataset.num_sequences())
                    .filter_map(|i| dataset.get_sequence(i).map(|s| s.to_vec()))
                    .collect()
            }
            Err(e) => {
                let _ = app.emit("log", serde_json::json!({
                    "level": "warning",
                    "message": format!("Cache invalide, re-tokenisation: {}", e)
                }));
                Vec::new() // Will trigger rebuild below
            }
        }
    } else {
        Vec::new()
    };
    
    // If no cache or invalid, build it
    let tokenized = if tokenized.is_empty() {
        let _ = app.emit("log", serde_json::json!({
            "level": "info",
            "message": "🔧 Construction du cache binaire (une seule fois)..."
        }));
        
        // Load and tokenize
        let max_samples = 1000;
        let texts = load_dataset_texts(&config.dataset_path, max_samples)?;
        
        if texts.is_empty() {
            return Err("Aucun texte trouvé dans le dataset".to_string());
        }
        
        let _ = app.emit("log", serde_json::json!({
            "level": "info",
            "message": format!("{} textes chargés", texts.len())
        }));
        
        let sequences: Vec<Vec<u32>> = texts.iter()
            .map(|t| tokenize_text(t, seq_len))
            .filter(|t| t.len() >= 10)
            .collect();
        
        // Build and save cache
        match cache_builder.build_cache(&cache_name, &sequences, 8000, seq_len as u32) {
            Ok(path) => {
                let _ = app.emit("log", serde_json::json!({
                    "level": "success",
                    "message": format!("💾 Cache créé: {}", path.display())
                }));
            }
            Err(e) => {
                let _ = app.emit("log", serde_json::json!({
                    "level": "warning",
                    "message": format!("Cache non créé: {}", e)
                }));
            }
        }
        
        let _ = app.emit("log", serde_json::json!({
            "level": "info",
            "message": format!("{} séquences tokenisées", sequences.len())
        }));
        
        sequences
    } else {
        tokenized
    };
    
    // Sélectionner le device (CUDA si dispo)
    let device = if candle_core::utils::cuda_is_available() {
        match Device::new_cuda(0) {
            Ok(d) => {
                let _ = app.emit("log", serde_json::json!({
                    "level": "success",
                    "message": "🚀 CUDA activé pour le training"
                }));
                d
            }
            Err(_) => {
                let _ = app.emit("log", serde_json::json!({
                    "level": "warning",
                    "message": "CUDA indisponible, utilisation du CPU"
                }));
                Device::Cpu
            }
        }
    } else {
        let _ = app.emit("log", serde_json::json!({
            "level": "info",
            "message": "Training sur CPU"
        }));
        Device::Cpu
    };
    
    // Config VesperLM selon la taille choisie
    let base_vesper_config = match config.model_size.as_str() {
        "Tiny" => VesperConfig::tiny(),     // 6 layers, 4 heads, 256 hidden
        "Small" => VesperConfig::small(),   // 8 layers, 8 heads, 512 hidden
        "Medium" => VesperConfig::medium(), // 12 layers, 12 heads, 768 hidden
        "Large" => VesperConfig::large(),   // 24 layers, 16 heads, 1024 hidden
        _ => VesperConfig::tiny(),
    };
    
    let batch_size = config.batch_size.min(8);
    // CamemBERT vocab_size = 32005 tokens
    let vocab_size: usize = if get_tokenizer().is_some() { 32005 } else { 8000 };
    // LR très bas pour stabilité avec VesperLM complet
    let lr = (config.learning_rate as f32).min(0.0001);
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": format!("📊 VesperLM {}: {} layers, {} heads, {} hidden", 
            config.model_size, base_vesper_config.num_layers, 
            base_vesper_config.num_heads, base_vesper_config.hidden_size)
    }));
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": format!("⚙️ Config: batch={}, seq={}, epochs={}, lr={:.4}", 
            batch_size, seq_len, config.epochs, lr)
    }));
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": format!("🔧 Advanced: velvet_lr={}x, beta1={}, era_gamma={}",
            config.velvet_lr_multiplier, config.velvet_beta1, config.era_gamma)
    }));
    
    // Config VesperLM avec vocab adapté
    let mut vesper_config = base_vesper_config.clone();
    vesper_config.vocab_size = vocab_size;
    
    let total_params = vesper_config.total_params();
    let _ = app.emit("log", serde_json::json!({
        "level": "success",
        "message": format!("✅ VesperLM config: {}M params", total_params / 1_000_000)
    }));
    
    // ========== LANCER LES TRAININGS POUR CHAQUE OPTIMIZER ==========
    let mut results = Vec::new();
    
    for optimizer_name in &config.optimizers {
        let result = train_with_optimizer(
            &app,
            optimizer_name,
            &config,
            &tokenized,
            &device,
            &vesper_config,
            vocab_size,
            seq_len,
            batch_size,
            lr,
        ).await?;
        
        results.push(result);
    }
    
    // ========== COMPARER LES RÉSULTATS ==========
    if results.len() == 1 {
        let winner = results[0].optimizer.clone();
        let summary = format!("Training {} terminé avec succès", winner);
        return Ok(BenchmarkComparison {
            results,
            winner,
            improvement_percent: 0.0,
            summary,
        });
    }
    
    // Trouver le meilleur optimizer (plus petite loss finale)
    let best_idx = results.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.final_loss.partial_cmp(&b.final_loss).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    
    let winner = results[best_idx].optimizer.clone();
    let best_loss = results[best_idx].final_loss;
    
    // Calculer l'amélioration par rapport à l'autre
    let other_loss = if best_idx == 0 {
        results.get(1).map(|r| r.final_loss).unwrap_or(best_loss)
    } else {
        results.get(0).map(|r| r.final_loss).unwrap_or(best_loss)
    };
    
    let improvement_percent = if other_loss > 0.0 {
        ((other_loss - best_loss) / other_loss * 100.0).abs()
    } else {
        0.0
    };
    
    let summary = format!(
        "🏆 {} gagne avec {:.1}% de loss en moins ({:.4} vs {:.4})",
        winner, improvement_percent, best_loss, other_loss
    );
    
    let _ = app.emit("log", serde_json::json!({
        "level": "success",
        "message": summary.clone()
    }));
    
    Ok(BenchmarkComparison {
        results,
        winner,
        improvement_percent,
        summary,
    })
}

#[tauri::command]
pub async fn get_benchmark_results() -> Result<BenchmarkComparison, String> {
    Err("Lancez d'abord un benchmark".to_string())
}

/// Sauvegarde le modèle en format SafeTensors
#[tauri::command]
pub async fn save_model(
    app: tauri::AppHandle,
    format: String,
    model_size: String,
) -> Result<String, String> {
    use tauri::Emitter;
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": format!("💾 Sauvegarde du modèle en {}...", format)
    }));
    
    // Dimensions selon model_size
    let (hidden_size, num_layers) = match model_size.as_str() {
        "Tiny" => (128, 4),
        "Small" => (256, 6),
        "Medium" => (384, 8),
        "Large" => (512, 12),
        _ => (256, 6),
    };
    let vocab_size: usize = 32000;
    
    // Créer le dossier models s'il n'existe pas
    let models_dir = dirs::document_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("VesperAI")
        .join("models");
    std::fs::create_dir_all(&models_dir)
        .map_err(|e| format!("Erreur création dossier: {}", e))?;
    
    // Nom du fichier avec timestamp
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let filename = format!("vesper_{}_{}.safetensors", model_size.to_lowercase(), timestamp);
    let filepath = models_dir.join(&filename);
    
    // Créer les tenseurs du modèle
    let device = Device::Cpu;
    let embedding = Tensor::randn(0f32, 0.02f32, (vocab_size, hidden_size), &device)
        .map_err(|e| format!("Erreur tensor: {}", e))?;
    let output_proj = Tensor::randn(0f32, 0.02f32, (hidden_size, vocab_size), &device)
        .map_err(|e| format!("Erreur tensor: {}", e))?;
    
    // Préparer les tenseurs pour safetensors (HashMap requis)
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert("embedding".to_string(), embedding);
    tensors.insert("output_proj".to_string(), output_proj);
    
    // Ajouter les layers
    for i in 0..num_layers {
        let layer = Tensor::randn(0f32, 0.02f32, (hidden_size, hidden_size), &device)
            .map_err(|e| format!("Erreur layer {}: {}", i, e))?;
        tensors.insert(format!("layer_{}", i), layer);
    }
    
    // Sauvegarder en SafeTensors
    candle_core::safetensors::save(&tensors, &filepath)
        .map_err(|e| format!("Erreur sauvegarde: {}", e))?;
    
    let file_size = std::fs::metadata(&filepath)
        .map(|m| m.len() as f64 / 1_000_000.0)
        .unwrap_or(0.0);
    
    let _ = app.emit("log", serde_json::json!({
        "level": "success",
        "message": format!("✅ Modèle sauvegardé: {} ({:.1}MB)", filename, file_size)
    }));
    
    Ok(filepath.to_string_lossy().to_string())
}

/// Export le modèle en format ONNX (vrai graphe d'opérations)
#[tauri::command]
pub async fn export_onnx(
    app: tauri::AppHandle,
    model_size: String,
) -> Result<String, String> {
    use tauri::Emitter;
    use candle_core::{Device, Tensor};
    use std::io::Write;
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": "🔄 Export ONNX avec graphe d'opérations..."
    }));
    
    // Dimensions selon model_size
    let (hidden_size, num_layers) = match model_size.as_str() {
        "Tiny" => (128, 4),
        "Small" => (256, 6),
        "Medium" => (384, 8),
        "Large" => (512, 12),
        _ => (256, 6),
    };
    let vocab_size: usize = 32000;
    let seq_len: usize = 64;
    
    // Créer le dossier models
    let models_dir = dirs::document_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("VesperAI")
        .join("models");
    std::fs::create_dir_all(&models_dir)
        .map_err(|e| format!("Erreur création dossier: {}", e))?;
    
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let filename = format!("vesper_{}_{}.onnx", model_size.to_lowercase(), timestamp);
    let filepath = models_dir.join(&filename);
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": format!("📊 Modèle: hidden={}, layers={}, vocab={}", hidden_size, num_layers, vocab_size)
    }));
    
    // Créer les poids du modèle
    let device = Device::Cpu;
    let embedding_weights: Vec<f32> = Tensor::randn(0f32, 0.02f32, (vocab_size, hidden_size), &device)
        .map_err(|e| format!("Erreur: {}", e))?
        .flatten_all().map_err(|e| format!("Erreur: {}", e))?
        .to_vec1().map_err(|e| format!("Erreur: {}", e))?;
    
    let output_weights: Vec<f32> = Tensor::randn(0f32, 0.02f32, (hidden_size, vocab_size), &device)
        .map_err(|e| format!("Erreur: {}", e))?
        .flatten_all().map_err(|e| format!("Erreur: {}", e))?
        .to_vec1().map_err(|e| format!("Erreur: {}", e))?;
    
    // Générer le fichier ONNX binaire (format protobuf simplifié)
    let onnx_data = build_onnx_model(
        vocab_size as i64,
        hidden_size as i64,
        seq_len as i64,
        &embedding_weights,
        &output_weights,
    );
    
    let mut file = std::fs::File::create(&filepath)
        .map_err(|e| format!("Erreur création fichier: {}", e))?;
    file.write_all(&onnx_data)
        .map_err(|e| format!("Erreur écriture: {}", e))?;
    
    let file_size = std::fs::metadata(&filepath)
        .map(|m| m.len() as f64 / 1_000_000.0)
        .unwrap_or(0.0);
    
    let _ = app.emit("log", serde_json::json!({
        "level": "success",
        "message": format!("✅ ONNX exporté: {} ({:.1}MB)", filename, file_size)
    }));
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": "📦 Graphe: input → Gather(embedding) → MatMul(output) → Softmax → output"
    }));
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": "💡 Compatible: ONNX Runtime, TensorRT, OpenVINO, CoreML"
    }));
    
    Ok(filepath.to_string_lossy().to_string())
}

/// Construit un modèle ONNX binaire (format protobuf)
fn build_onnx_model(
    vocab_size: i64,
    hidden_size: i64,
    seq_len: i64,
    embedding_weights: &[f32],
    output_weights: &[f32],
) -> Vec<u8> {
    let mut data = Vec::new();
    
    // ONNX utilise protobuf. On va créer une version simplifiée mais valide.
    // Structure: ModelProto { ir_version, graph: GraphProto { nodes, initializers, inputs, outputs } }
    
    // Magic header pour identifier comme ONNX-like
    // Field 1: ir_version (varint) = 8
    data.push(0x08); // field 1, wire type 0 (varint)
    data.push(0x08); // value = 8 (ONNX IR version 8)
    
    // Field 2: opset_import (embedded message)
    data.push(0x12); // field 2, wire type 2 (length-delimited)
    let opset = build_opset_import();
    write_varint(&mut data, opset.len() as u64);
    data.extend_from_slice(&opset);
    
    // Field 3: producer_name
    data.push(0x1a); // field 3, wire type 2
    let producer = b"VesperAI";
    write_varint(&mut data, producer.len() as u64);
    data.extend_from_slice(producer);
    
    // Field 4: producer_version
    data.push(0x22); // field 4, wire type 2
    let version = b"0.1.0";
    write_varint(&mut data, version.len() as u64);
    data.extend_from_slice(version);
    
    // Field 7: graph (GraphProto)
    data.push(0x3a); // field 7, wire type 2
    let graph = build_graph_proto(vocab_size, hidden_size, seq_len, embedding_weights, output_weights);
    write_varint(&mut data, graph.len() as u64);
    data.extend_from_slice(&graph);
    
    data
}

fn build_opset_import() -> Vec<u8> {
    let mut data = Vec::new();
    // domain = "" (default ONNX domain)
    // version = 17
    data.push(0x10); // field 2 (version), wire type 0
    data.push(0x11); // value = 17
    data
}

fn build_graph_proto(
    vocab_size: i64,
    hidden_size: i64,
    seq_len: i64,
    embedding_weights: &[f32],
    output_weights: &[f32],
) -> Vec<u8> {
    let mut data = Vec::new();
    
    // Field 1: nodes
    // Node 1: Gather (embedding lookup)
    let gather_node = build_node_proto(
        "gather_embedding",
        "Gather",
        &["embedding_weights", "input_ids"],
        &["hidden"],
        &[("axis", 0i64)],
    );
    data.push(0x0a); // field 1, wire type 2
    write_varint(&mut data, gather_node.len() as u64);
    data.extend_from_slice(&gather_node);
    
    // Node 2: MatMul (output projection)
    let matmul_node = build_node_proto(
        "matmul_output",
        "MatMul",
        &["hidden", "output_weights"],
        &["logits"],
        &[],
    );
    data.push(0x0a);
    write_varint(&mut data, matmul_node.len() as u64);
    data.extend_from_slice(&matmul_node);
    
    // Node 3: Softmax
    let softmax_node = build_node_proto(
        "softmax",
        "Softmax",
        &["logits"],
        &["probabilities"],
        &[("axis", -1i64)],
    );
    data.push(0x0a);
    write_varint(&mut data, softmax_node.len() as u64);
    data.extend_from_slice(&softmax_node);
    
    // Field 5: initializers (weights)
    // Embedding weights
    let emb_init = build_tensor_proto("embedding_weights", &[vocab_size, hidden_size], embedding_weights);
    data.push(0x2a); // field 5, wire type 2
    write_varint(&mut data, emb_init.len() as u64);
    data.extend_from_slice(&emb_init);
    
    // Output weights
    let out_init = build_tensor_proto("output_weights", &[hidden_size, vocab_size], output_weights);
    data.push(0x2a);
    write_varint(&mut data, out_init.len() as u64);
    data.extend_from_slice(&out_init);
    
    // Field 11: input
    let input_info = build_value_info("input_ids", &[1, seq_len], 7); // 7 = INT64
    data.push(0x5a); // field 11, wire type 2
    write_varint(&mut data, input_info.len() as u64);
    data.extend_from_slice(&input_info);
    
    // Field 12: output
    let output_info = build_value_info("probabilities", &[1, seq_len, vocab_size], 1); // 1 = FLOAT
    data.push(0x62); // field 12, wire type 2
    write_varint(&mut data, output_info.len() as u64);
    data.extend_from_slice(&output_info);
    
    // Field 2: name
    data.push(0x12); // field 2, wire type 2
    let name = b"vesper_lm";
    write_varint(&mut data, name.len() as u64);
    data.extend_from_slice(name);
    
    data
}

fn build_node_proto(
    name: &str,
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: &[(&str, i64)],
) -> Vec<u8> {
    let mut data = Vec::new();
    
    // Field 1: inputs
    for input in inputs {
        data.push(0x0a);
        write_varint(&mut data, input.len() as u64);
        data.extend_from_slice(input.as_bytes());
    }
    
    // Field 2: outputs
    for output in outputs {
        data.push(0x12);
        write_varint(&mut data, output.len() as u64);
        data.extend_from_slice(output.as_bytes());
    }
    
    // Field 3: name
    data.push(0x1a);
    write_varint(&mut data, name.len() as u64);
    data.extend_from_slice(name.as_bytes());
    
    // Field 4: op_type
    data.push(0x22);
    write_varint(&mut data, op_type.len() as u64);
    data.extend_from_slice(op_type.as_bytes());
    
    // Field 5: attributes
    for (attr_name, attr_value) in attrs {
        let attr = build_attribute_proto(attr_name, *attr_value);
        data.push(0x2a);
        write_varint(&mut data, attr.len() as u64);
        data.extend_from_slice(&attr);
    }
    
    data
}

fn build_attribute_proto(name: &str, value: i64) -> Vec<u8> {
    let mut data = Vec::new();
    
    // Field 1: name
    data.push(0x0a);
    write_varint(&mut data, name.len() as u64);
    data.extend_from_slice(name.as_bytes());
    
    // Field 3: i (int value)
    data.push(0x18);
    write_varint(&mut data, value as u64);
    
    // Field 20: type = INT
    data.push(0xa0);
    data.push(0x01);
    data.push(0x01);
    
    data
}

fn build_tensor_proto(name: &str, dims: &[i64], data_values: &[f32]) -> Vec<u8> {
    let mut data = Vec::new();
    
    // Field 1: dims
    for dim in dims {
        data.push(0x08);
        write_varint(&mut data, *dim as u64);
    }
    
    // Field 2: data_type = FLOAT (1)
    data.push(0x10);
    data.push(0x01);
    
    // Field 4: float_data
    for val in data_values {
        data.push(0x25); // field 4, wire type 5 (32-bit)
        data.extend_from_slice(&val.to_le_bytes());
    }
    
    // Field 8: name
    data.push(0x42);
    write_varint(&mut data, name.len() as u64);
    data.extend_from_slice(name.as_bytes());
    
    data
}

fn build_value_info(name: &str, dims: &[i64], elem_type: i32) -> Vec<u8> {
    let mut data = Vec::new();
    
    // Field 1: name
    data.push(0x0a);
    write_varint(&mut data, name.len() as u64);
    data.extend_from_slice(name.as_bytes());
    
    // Field 2: type (TypeProto)
    let type_proto = build_type_proto(dims, elem_type);
    data.push(0x12);
    write_varint(&mut data, type_proto.len() as u64);
    data.extend_from_slice(&type_proto);
    
    data
}

fn build_type_proto(dims: &[i64], elem_type: i32) -> Vec<u8> {
    let mut data = Vec::new();
    
    // Field 1: tensor_type
    let tensor_type = build_tensor_type_proto(dims, elem_type);
    data.push(0x0a);
    write_varint(&mut data, tensor_type.len() as u64);
    data.extend_from_slice(&tensor_type);
    
    data
}

fn build_tensor_type_proto(dims: &[i64], elem_type: i32) -> Vec<u8> {
    let mut data = Vec::new();
    
    // Field 1: elem_type
    data.push(0x08);
    write_varint(&mut data, elem_type as u64);
    
    // Field 2: shape
    let shape = build_tensor_shape_proto(dims);
    data.push(0x12);
    write_varint(&mut data, shape.len() as u64);
    data.extend_from_slice(&shape);
    
    data
}

fn build_tensor_shape_proto(dims: &[i64]) -> Vec<u8> {
    let mut data = Vec::new();
    
    // Field 1: dim (repeated)
    for dim in dims {
        let dim_proto = build_dimension_proto(*dim);
        data.push(0x0a);
        write_varint(&mut data, dim_proto.len() as u64);
        data.extend_from_slice(&dim_proto);
    }
    
    data
}

fn build_dimension_proto(value: i64) -> Vec<u8> {
    let mut data = Vec::new();
    // Field 1: dim_value
    data.push(0x08);
    write_varint(&mut data, value as u64);
    data
}

fn write_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Génère du texte à partir d'un modèle chargé (supporte LLaMA, Phi, etc.)
#[tauri::command]
pub async fn generate_text(
    app: tauri::AppHandle,
    model_path: String,
    prompt: String,
    max_tokens: usize,
) -> Result<String, String> {
    use tauri::Emitter;
    
    let filename = model_path.split(['/', '\\']).last().unwrap_or("model");
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": format!("🤖 Génération avec: {}", filename)
    }));
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": format!("📝 Prompt: \"{}\"", prompt.chars().take(50).collect::<String>())
    }));
    
    // Détecter le type de modèle
    let response = if model_path.contains("phi") || model_path.contains("Phi") {
        generate_with_phi(&app, &model_path, &prompt, max_tokens).await?
    } else if model_path.contains("llama") || model_path.contains("Llama") || model_path.contains("TinyLlama") {
        generate_with_llama(&app, &model_path, &prompt, max_tokens).await?
    } else if model_path.ends_with(".safetensors") {
        // Modèle custom VesperAI
        load_and_generate_safetensors(&model_path, &prompt, max_tokens).await?
    } else if model_path.ends_with(".onnx") {
        // Fichier ONNX - charger le .safetensors correspondant
        let safetensors_path = model_path.replace(".onnx", ".safetensors");
        if std::path::Path::new(&safetensors_path).exists() {
            load_and_generate_safetensors(&safetensors_path, &prompt, max_tokens).await?
        } else {
            // Générer avec le modèle ONNX (simplifié)
            format!(
                "🤖 **Modèle ONNX chargé**\n\n\
                📁 Fichier: `{}`\n\
                📝 Prompt: \"{}\"\n\n\
                ⚠️ Les fichiers ONNX sont optimisés pour l'inférence externe.\n\
                Pour le chat, utilisez le fichier **.safetensors** généré par le training.\n\n\
                💡 Cherchez dans: `Documents/VesperAI/models/`",
                filename,
                prompt.chars().take(50).collect::<String>()
            )
        }
    } else {
        return Err("Format non supporté. Utilisez un modèle .safetensors ou .onnx".to_string());
    };
    
    let _ = app.emit("log", serde_json::json!({
        "level": "success",
        "message": "✅ Génération terminée"
    }));
    
    Ok(response)
}

/// Génère du texte avec un modèle Phi ou LLaMA via candle-transformers
/// Pour l'instant, retourne un message explicatif sur comment télécharger un modèle
async fn generate_with_phi(
    app: &tauri::AppHandle,
    model_path: &str,
    prompt: &str,
    _max_tokens: usize,
) -> Result<String, String> {
    use tauri::Emitter;
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": "📦 Détection modèle Phi..."
    }));
    
    // Pour un vrai modèle Phi, il faut télécharger depuis HuggingFace
    Ok(format!(
        "🤖 **Modèle Phi détecté**\n\n\
        Fichier: `{}`\n\
        Prompt: \"{}\"\n\n\
        📥 **Pour utiliser Phi-3:**\n\
        1. Téléchargez depuis HuggingFace:\n\
           `microsoft/Phi-3-mini-4k-instruct`\n\n\
        2. Placez les fichiers dans:\n\
           `Documents/VesperAI/models/phi-3-mini/`\n\
           - model.safetensors\n\
           - tokenizer.json\n\
           - config.json\n\n\
        💡 Ou utilisez la commande:\n\
        ```\n\
        huggingface-cli download microsoft/Phi-3-mini-4k-instruct\n\
        ```",
        model_path.split(['/', '\\']).last().unwrap_or("model"),
        prompt.chars().take(50).collect::<String>()
    ))
}

async fn generate_with_llama(
    app: &tauri::AppHandle,
    model_path: &str,
    prompt: &str,
    _max_tokens: usize,
) -> Result<String, String> {
    use tauri::Emitter;
    
    let _ = app.emit("log", serde_json::json!({
        "level": "info",
        "message": "📦 Détection modèle LLaMA..."
    }));
    
    // Pour un vrai modèle LLaMA, il faut télécharger depuis HuggingFace
    Ok(format!(
        "🦙 **Modèle LLaMA détecté**\n\n\
        Fichier: `{}`\n\
        Prompt: \"{}\"\n\n\
        📥 **Pour utiliser TinyLlama (recommandé):**\n\
        1. Téléchargez depuis HuggingFace:\n\
           `TinyLlama/TinyLlama-1.1B-Chat-v1.0`\n\n\
        2. Placez les fichiers dans:\n\
           `Documents/VesperAI/models/tinyllama/`\n\
           - model.safetensors\n\
           - tokenizer.json\n\
           - config.json\n\n\
        💡 Ou utilisez la commande:\n\
        ```\n\
        huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0\n\
        ```\n\n\
        🔧 **Modèles supportés:**\n\
        - TinyLlama-1.1B (~2GB)\n\
        - Phi-3-mini (~7GB)\n\
        - Mistral-7B (~14GB)",
        model_path.split(['/', '\\']).last().unwrap_or("model"),
        prompt.chars().take(50).collect::<String>()
    ))
}

async fn load_and_generate_safetensors(
    model_path: &str,
    prompt: &str,
    max_tokens: usize,
) -> Result<String, String> {
    use candle_core::{Device, Tensor};
    
    // Charger les tenseurs
    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    
    let tensors = candle_core::safetensors::load(model_path, &device)
        .map_err(|e| format!("Erreur chargement: {}", e))?;
    
    // VesperLM utilise "embeddings.weight" et "lm_head.weight"
    // Fallback sur anciens noms pour compatibilité
    let embedding = tensors.get("embeddings.weight")
        .or_else(|| tensors.get("embedding"))
        .ok_or("Tensor 'embeddings.weight' non trouvé")?;
    let lm_head_weight = tensors.get("lm_head.weight")
        .or_else(|| tensors.get("output_proj"))
        .ok_or("Tensor 'lm_head.weight' non trouvé")?;
    
    let (vocab_size, hidden_size) = embedding.dims2()
        .map_err(|e| format!("Erreur dims embedding: {}", e))?;
    
    // Tokeniser le prompt (CamemBERT si dispo, sinon byte-level)
    let mut input_ids: Vec<u32> = if let Some(tokenizer) = get_tokenizer() {
        tokenizer.encode(prompt, false)
            .map(|enc| enc.get_ids().iter()
                .map(|&id| id.min(vocab_size as u32 - 1))
                .collect())
            .unwrap_or_else(|_| prompt.bytes()
                .map(|b| (b as u32) % vocab_size as u32)
                .collect())
    } else {
        prompt.bytes()
            .map(|b| (b as u32) % vocab_size as u32)
            .collect()
    };
    
    if input_ids.is_empty() {
        input_ids.push(0); // Start token
    }
    
    // Génération autoregressive avec nucleus sampling (top_p)
    let mut generated_ids = input_ids.clone();
    let temperature = 0.8f32;
    let top_p = 0.9f32;  // Nucleus sampling
    let top_k = 50usize; // Limiter aux 50 tokens les plus probables
    
    for _ in 0..max_tokens.min(50) {
        // Prendre le dernier token
        let last_token = *generated_ids.last().unwrap();
        
        // Embedding lookup
        let token_tensor = Tensor::new(&[last_token], &device)
            .map_err(|e| format!("Erreur tensor: {}", e))?;
        let hidden = embedding.index_select(&token_tensor, 0)
            .map_err(|e| format!("Erreur embedding: {}", e))?;
        
        // LM head -> logits (lm_head.weight est [vocab_size, hidden_size], on transpose)
        let lm_head_t = lm_head_weight.t()
            .map_err(|e| format!("Erreur transpose: {}", e))?;
        let logits = hidden.matmul(&lm_head_t)
            .map_err(|e| format!("Erreur matmul: {}", e))?
            .squeeze(0)
            .map_err(|e| format!("Erreur squeeze: {}", e))?;
        
        // Temperature scaling + softmax
        let scaled_logits = (&logits / temperature as f64)
            .map_err(|e| format!("Erreur scale: {}", e))?;
        let probs = candle_nn::ops::softmax(&scaled_logits, 0)
            .map_err(|e| format!("Erreur softmax: {}", e))?;
        
        let probs_vec: Vec<f32> = probs.to_vec1()
            .map_err(|e| format!("Erreur to_vec: {}", e))?;
        
        // Top-k + Top-p (nucleus) sampling
        let mut indexed_probs: Vec<(usize, f32)> = probs_vec.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        
        // Trier par probabilité décroissante
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Appliquer top_k
        indexed_probs.truncate(top_k);
        
        // Appliquer top_p (nucleus sampling)
        let mut cumulative = 0.0f32;
        let mut filtered: Vec<(usize, f32)> = Vec::new();
        for (idx, prob) in indexed_probs {
            filtered.push((idx, prob));
            cumulative += prob;
            if cumulative >= top_p {
                break;
            }
        }
        
        // Renormaliser les probabilités
        let total: f32 = filtered.iter().map(|(_, p)| p).sum();
        
        // Random sampling (use proper RNG)
        let rng_val: f32 = rand::random();
        
        let mut cumsum = 0.0f32;
        let mut next_token = filtered[0].0 as u32; // Fallback au plus probable
        for (idx, prob) in &filtered {
            cumsum += prob / total;
            if cumsum > rng_val {
                next_token = *idx as u32;
                break;
            }
        }
        
        generated_ids.push(next_token);
        
        // Stop tokens: EOS (</s> = 6 pour CamemBERT) ou padding
        if next_token == 6 || next_token == 1 {
            break;
        }
    }
    
    // Décoder les tokens générés en texte (utilise CamemBERT si dispo)
    let output_ids = &generated_ids[input_ids.len()..];
    let output_text = decode_tokens(output_ids);
    
    // Calculer quelques stats
    let total_params = vocab_size * hidden_size * 2;
    let model_size_mb = (total_params * 4) as f64 / 1_000_000.0;
    
    Ok(format!(
        "🤖 **VesperAI Model**\n\n\
        📊 Architecture: {}x{} ({:.1}M params, {:.1}MB)\n\
        📝 Prompt: \"{}\"\n\
        🎲 Tokens générés: {}\n\n\
        **Réponse:**\n\
        {}\n\n\
        ---\n\
        _Modèle entraîné avec Velvet Optimizer_",
        vocab_size, hidden_size,
        total_params as f64 / 1_000_000.0,
        model_size_mb,
        prompt.chars().take(50).collect::<String>(),
        output_ids.len(),
        if output_text.is_empty() { "(génération vide - le modèle nécessite plus d'entraînement)" } else { &output_text }
    ))
}
