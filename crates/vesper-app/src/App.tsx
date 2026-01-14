import { useState, useEffect, useRef } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'
import { open } from '@tauri-apps/plugin-dialog'
import { Play, Square, Settings, Database, Cpu, Zap, Sparkles, Search, Download, BarChart3, Cloud, FolderOpen, FlaskConical, MessageSquare, X, Bot, User, Send } from 'lucide-react'
import Aurora from './components/reactbits/Aurora'
import GradientText from './components/reactbits/GradientText'
import SpotlightCard from './components/reactbits/SpotlightCard'

interface TrainingStatus {
  is_running: boolean
  current_epoch: number
  total_epochs: number
  current_loss: number
  progress: number
}

interface ModelConfig {
  name: string
  params: string
  hidden_size: number
  num_layers: number
}

interface HFDataset {
  id: string
  description: string
  downloads: number
  likes: number
  tags: string[]
}

interface BenchmarkResult {
  optimizer: string
  final_loss: number
  best_loss: number
  training_time_ms: number
  loss_history: number[]
  memory_peak_mb: number
  convergence_epoch: number | null
}

interface BenchmarkComparison {
  results: BenchmarkResult[]
  winner: string
  improvement_percent: number
  summary: string
}

type DatasetSource = 'local' | 'huggingface'

function App() {
  const [status, setStatus] = useState<TrainingStatus>({
    is_running: false,
    current_epoch: 0,
    total_epochs: 0,
    current_loss: 0,
    progress: 0,
  })

  const [selectedModel, setSelectedModel] = useState('Medium')
  const [epochs, setEpochs] = useState(3)
  const [datasetPath, setDatasetPath] = useState('')
  const [modelConfigs, setModelConfigs] = useState<ModelConfig[]>([])
  
  // New states
  const [datasetSource, setDatasetSource] = useState<DatasetSource>('local')
  const [hfSearchQuery, setHfSearchQuery] = useState('')
  const [hfDatasets, setHfDatasets] = useState<HFDataset[]>([])
  const [hfLoading, setHfLoading] = useState(false)
  const [selectedHfDataset, setSelectedHfDataset] = useState<string | null>(null)
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkComparison | null>(null)
  const [benchmarkRunning, setBenchmarkRunning] = useState(false)
  
  // Progression téléchargement
  const [dlProgress, setDlProgress] = useState<{
    status: string
    progress: number
    downloaded: number
    total: number
  } | null>(null)
  
  // Logs
  const [logs, setLogs] = useState<{ level: string; message: string; time: string }[]>([])
  const logsEndRef = useRef<HTMLDivElement>(null)
  
  // CUDA
  const [cudaInfo, setCudaInfo] = useState<{ available: boolean; device_name: string } | null>(null)
  
  // Advanced Settings
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [advancedConfig, setAdvancedConfig] = useState({
    velvet_lr_multiplier: 1.5,
    velvet_beta1: 0.95,
    era_temperature: 1.0,
    flylora_rank: 16,
    flylora_sparsity: 0.75,
  })
  
  // Chat Interface
  const [showChat, setShowChat] = useState(false)
  const [chatMessages, setChatMessages] = useState<{ role: 'user' | 'assistant'; content: string }[]>([])
  const [chatInput, setChatInput] = useState('')
  const [loadedModelPath, setLoadedModelPath] = useState<string | null>(null)
  const [generating, setGenerating] = useState(false)

  useEffect(() => {
    loadModelConfigs()
    checkCuda()
    
    // Charger config avancée depuis localStorage
    const savedConfig = localStorage.getItem('vesper_advanced_config')
    if (savedConfig) {
      try {
        setAdvancedConfig(JSON.parse(savedConfig))
      } catch (e) {
        console.error('Failed to parse saved config:', e)
      }
    }
    
    const interval = setInterval(updateStatus, 1000)
    
    // Écouter les événements de progression
    const unlistenDl = listen<{ status: string; progress: number; downloaded: number; total: number }>('dl-progress', (event) => {
      setDlProgress(event.payload)
      if (event.payload.progress >= 100) {
        setTimeout(() => setDlProgress(null), 2000)
      }
    })
    
    // Écouter les logs (TOUT garder, pas de limite)
    const unlistenLog = listen<{ level: string; message: string }>('log', (event) => {
      const time = new Date().toLocaleTimeString('fr-FR')
      setLogs(prev => [...prev, { ...event.payload, time }])
    })
    
    return () => {
      clearInterval(interval)
      unlistenDl.then(fn => fn())
      unlistenLog.then(fn => fn())
    }
  }, [])

  // Auto-scroll console vers le bas à chaque nouveau log
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const checkCuda = async () => {
    try {
      const info = await invoke<{ available: boolean; device_count: number; device_name: string }>('check_cuda')
      setCudaInfo(info)
    } catch (error) {
      setCudaInfo({ available: false, device_name: 'Erreur détection' })
    }
  }

  const loadModelConfigs = async () => {
    try {
      const configs = await invoke<ModelConfig[]>('get_model_configs')
      setModelConfigs(configs)
    } catch (error) {
      console.error('Failed to load model configs:', error)
      setModelConfigs([
        { name: 'Small', params: '125M', hidden_size: 768, num_layers: 12 },
        { name: 'Medium', params: '350M', hidden_size: 1024, num_layers: 24 },
        { name: 'Large', params: '760M', hidden_size: 1536, num_layers: 24 },
      ])
    }
  }

  const updateStatus = async () => {
    try {
      const newStatus = await invoke<TrainingStatus>('get_training_status')
      setStatus(newStatus)
    } catch (error) {
      // Silent fail for status updates
    }
  }

  const selectDataset = async () => {
    try {
      const selected = await open({
        filters: [{ name: 'Dataset', extensions: ['jsonl', 'json', 'txt'] }],
      })
      if (selected && typeof selected === 'string') {
        setDatasetPath(selected)
        // Charger et analyser le dataset
        try {
          const info = await invoke<{ path: string; num_samples: number; total_tokens: number }>('load_dataset', {
            path: selected
          })
          console.log(`Dataset chargé: ${info.num_samples} samples, ${info.total_tokens} tokens`)
        } catch (loadError) {
          console.error('Erreur chargement dataset:', loadError)
        }
      }
    } catch (error) {
      console.error('Erreur sélection fichier:', error)
    }
  }

  const startTraining = async () => {
    if (!datasetPath) return
    setBenchmarkRunning(true)
    try {
      const results = await invoke<BenchmarkComparison>('start_benchmark', {
        config: {
          dataset_path: datasetPath,
          model_size: selectedModel,
          epochs: epochs,
          batch_size: 8,
          learning_rate: 0.0001,
          optimizers: ['velvet'],
          velvet_lr_multiplier: advancedConfig.velvet_lr_multiplier,
          velvet_beta1: advancedConfig.velvet_beta1,
          era_temperature: advancedConfig.era_temperature,
          flylora_rank: advancedConfig.flylora_rank,
          flylora_sparsity: advancedConfig.flylora_sparsity,
        }
      })
      setBenchmarkResults(results)
    } catch (error) {
      console.error('Failed to start training:', error)
      setLogs(prev => [...prev, {
        time: new Date().toLocaleTimeString('fr-FR'),
        level: 'error',
        message: `❌ Erreur: ${error}`
      }])
    } finally {
      setBenchmarkRunning(false)
    }
  }

  const stopTraining = async () => {
    try {
      await invoke('stop_training')
    } catch (error) {
      console.error('Failed to stop training:', error)
    }
  }

  // HuggingFace Dataset Search
  const searchHuggingFace = async () => {
    if (!hfSearchQuery.trim()) return
    setHfLoading(true)
    try {
      const results = await invoke<HFDataset[]>('search_hf_datasets', { query: hfSearchQuery, limit: 8 })
      setHfDatasets(results)
    } catch (error) {
      console.error('Failed to search HuggingFace:', error)
    } finally {
      setHfLoading(false)
    }
  }

  const selectHfDataset = async (datasetId: string) => {
    setSelectedHfDataset(datasetId)
    setHfLoading(true)
    try {
      // Télécharge vraiment le dataset via l'API HuggingFace
      const info = await invoke<{ path: string; num_samples: number; total_tokens: number }>('load_hf_dataset', {
        datasetId,
        split: 'train',
        textColumn: null
      })
      setDatasetPath(info.path)
      console.log(`Dataset téléchargé: ${info.num_samples} samples, ${info.total_tokens} tokens`)
    } catch (error) {
      console.error('Failed to download dataset:', error)
      setDatasetPath(`hf://${datasetId}`) // Fallback
    } finally {
      setHfLoading(false)
    }
  }

  // Benchmark - reçoit les résultats directement
  const runBenchmark = async () => {
    if (!datasetPath) return
    setBenchmarkRunning(true)
    setBenchmarkResults(null)
    try {
      const results = await invoke<BenchmarkComparison>('start_benchmark', {
        config: {
          dataset_path: datasetPath,
          model_size: selectedModel,
          epochs: epochs,
          batch_size: 8,
          learning_rate: 0.0001,
          optimizers: ['velvet', 'adamw'],
          // Paramètres avancés
          velvet_lr_multiplier: advancedConfig.velvet_lr_multiplier,
          velvet_beta1: advancedConfig.velvet_beta1,
          era_temperature: advancedConfig.era_temperature,
          flylora_rank: advancedConfig.flylora_rank,
          flylora_sparsity: advancedConfig.flylora_sparsity,
        }
      })
      setBenchmarkResults(results)
    } catch (error) {
      console.error('Benchmark failed:', error)
      // Afficher l'erreur dans les logs
      setLogs(prev => [...prev, {
        time: new Date().toLocaleTimeString('fr-FR'),
        level: 'error',
        message: `❌ Erreur benchmark: ${error}`
      }])
    } finally {
      setBenchmarkRunning(false)
    }
  }

  const selectedConfig = modelConfigs.find((c) => c.name === selectedModel)

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Aurora Background */}
      <div className="fixed inset-0 z-0 opacity-60">
        <Aurora 
          colorStops={['#7c3aed', '#ec4899', '#7c3aed']} 
          amplitude={1.2} 
          speed={0.5}
          blend={0.6}
        />
      </div>

      {/* Content */}
      <div className="relative z-10 min-h-screen p-6 lg:p-8">
        {/* Header */}
        <header className="mb-8 flex items-center justify-between">
          <div>
            <GradientText 
              colors={['#a855f7', '#ec4899', '#f97316', '#a855f7']}
              animationSpeed={6}
              className="text-4xl lg:text-5xl font-bold"
            >
              VesperAI
            </GradientText>
            <p className="text-white/50 mt-2 text-sm">High-Performance LLM Training in Rust</p>
          </div>
          <div className="flex items-center gap-3">
            {cudaInfo && (
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
                cudaInfo.available 
                  ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
                  : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
              }`}>
                <div className={`w-2 h-2 rounded-full ${cudaInfo.available ? 'bg-green-400 animate-pulse' : 'bg-yellow-400'}`} />
                {cudaInfo.available ? '🚀 CUDA' : '💻 CPU'}
              </div>
            )}
            <div className="flex items-center gap-2 text-white/40 text-xs">
              <Zap className="w-4 h-4 text-violet-400" />
              <span>Candle</span>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Panel - Config */}
          <div className="lg:col-span-4 space-y-4">
            {/* Model Selection */}
            <SpotlightCard className="p-5" spotlightColor="rgba(168, 85, 247, 0.15)">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-lg bg-violet-500/20">
                  <Cpu className="w-5 h-5 text-violet-400" />
                </div>
                <h2 className="text-lg font-semibold">Model</h2>
              </div>
              
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-white/5 rounded-xl px-4 py-3 border border-white/10 focus:outline-none focus:border-violet-500/50 transition-colors text-sm"
              >
                {modelConfigs.map((config) => (
                  <option key={config.name} value={config.name} className="bg-black">
                    {config.name} ({config.params})
                  </option>
                ))}
              </select>

              {selectedConfig && (
                <div className="mt-4 grid grid-cols-2 gap-3">
                  <div className="bg-white/5 rounded-lg p-3">
                    <div className="text-xs text-white/40">Hidden Size</div>
                    <div className="text-lg font-semibold text-violet-300">{selectedConfig.hidden_size}</div>
                  </div>
                  <div className="bg-white/5 rounded-lg p-3">
                    <div className="text-xs text-white/40">Layers</div>
                    <div className="text-lg font-semibold text-pink-300">{selectedConfig.num_layers}</div>
                  </div>
                </div>
              )}
            </SpotlightCard>

            {/* Dataset */}
            <SpotlightCard className="p-5" spotlightColor="rgba(236, 72, 153, 0.15)">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-lg bg-pink-500/20">
                  <Database className="w-5 h-5 text-pink-400" />
                </div>
                <h2 className="text-lg font-semibold">Dataset</h2>
              </div>
              
              {/* Source Tabs */}
              <div className="flex gap-2 mb-4">
                <button
                  onClick={() => setDatasetSource('local')}
                  className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-xs font-medium transition-all ${
                    datasetSource === 'local' 
                      ? 'bg-white/10 text-white' 
                      : 'text-white/40 hover:text-white/60'
                  }`}
                >
                  <FolderOpen className="w-3.5 h-3.5" />
                  Local
                </button>
                <button
                  onClick={() => setDatasetSource('huggingface')}
                  className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-xs font-medium transition-all ${
                    datasetSource === 'huggingface' 
                      ? 'bg-white/10 text-white' 
                      : 'text-white/40 hover:text-white/60'
                  }`}
                >
                  <Cloud className="w-3.5 h-3.5" />
                  HuggingFace
                </button>
              </div>

              {datasetSource === 'local' ? (
                <button
                  onClick={selectDataset}
                  className="w-full bg-gradient-to-r from-violet-600 to-pink-600 hover:from-violet-500 hover:to-pink-500 rounded-xl px-4 py-3 transition-all text-sm font-medium"
                >
                  Select File
                </button>
              ) : (
                <div className="space-y-3">
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={hfSearchQuery}
                      onChange={(e) => setHfSearchQuery(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && searchHuggingFace()}
                      placeholder="Search datasets..."
                      className="flex-1 bg-white/5 rounded-lg px-3 py-2 border border-white/10 focus:outline-none focus:border-pink-500/50 text-sm"
                    />
                    <button
                      onClick={searchHuggingFace}
                      disabled={hfLoading}
                      className="p-2 bg-pink-600 hover:bg-pink-500 rounded-lg transition-colors"
                    >
                      <Search className="w-4 h-4" />
                    </button>
                  </div>
                  
                  {/* Indicateur de progression réel */}
                  {dlProgress && (
                    <div className="py-3 space-y-2">
                      <div className="flex justify-between text-xs">
                        <span className="text-white/70">{dlProgress.status}</span>
                        <span className="text-pink-400 font-medium">{dlProgress.progress}%</span>
                      </div>
                      <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
                        <div 
                          className="bg-gradient-to-r from-pink-500 to-violet-500 h-full rounded-full transition-all duration-300"
                          style={{ width: `${dlProgress.progress}%` }}
                        />
                      </div>
                      {dlProgress.total > 0 && (
                        <div className="text-xs text-white/50 text-center">
                          {dlProgress.downloaded.toLocaleString()} / {dlProgress.total.toLocaleString()} lignes
                        </div>
                      )}
                    </div>
                  )}
                  
                  {hfLoading && !dlProgress && (
                    <div className="flex items-center justify-center gap-2 py-4">
                      <div className="w-4 h-4 border-2 border-pink-500 border-t-transparent rounded-full animate-spin"></div>
                      <span className="text-xs text-white/60">Connexion...</span>
                    </div>
                  )}

                  {!hfLoading && hfDatasets.length > 0 && (
                    <div className="max-h-40 overflow-y-auto space-y-1">
                      {hfDatasets.map((ds) => (
                        <button
                          key={ds.id}
                          onClick={() => selectHfDataset(ds.id)}
                          disabled={hfLoading}
                          className={`w-full text-left p-2 rounded-lg text-xs transition-colors ${
                            selectedHfDataset === ds.id
                              ? 'bg-pink-500/20 border border-pink-500/50'
                              : 'bg-white/5 hover:bg-white/10'
                          } ${hfLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                          <div className="font-medium truncate">{ds.id}</div>
                          <div className="text-white/40 flex items-center gap-2 mt-0.5">
                            <Download className="w-3 h-3" />
                            {ds.downloads.toLocaleString()}
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
              
              {datasetPath && (
                <div className="mt-3 text-xs text-white/50 truncate bg-white/5 rounded-lg px-3 py-2">
                  {datasetPath.startsWith('hf://') ? datasetPath : datasetPath.split('\\').pop()}
                </div>
              )}
            </SpotlightCard>

            {/* Training Config */}
            <SpotlightCard className="p-5" spotlightColor="rgba(249, 115, 22, 0.15)">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-lg bg-orange-500/20">
                  <Settings className="w-5 h-5 text-orange-400" />
                </div>
                <h2 className="text-lg font-semibold">Training</h2>
              </div>
              
              <div className="space-y-3">
                <div>
                  <label className="block mb-2 text-xs text-white/50">Epochs</label>
                  <input
                    type="number"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
                    min="1"
                    max="100"
                    className="w-full bg-white/5 rounded-xl px-4 py-3 border border-white/10 focus:outline-none focus:border-orange-500/50 transition-colors text-sm"
                  />
                </div>
              </div>
            </SpotlightCard>
          </div>

          {/* Right Panel - Status & Control */}
          <div className="lg:col-span-8 space-y-4">
            {/* Control */}
            <SpotlightCard className="p-6" spotlightColor="rgba(34, 197, 94, 0.15)">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold mb-1">Training Control</h2>
                  <p className="text-white/50 text-sm">
                    {status.is_running ? (
                      <span className="flex items-center gap-2">
                        <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                        Training in progress...
                      </span>
                    ) : (
                      'Ready to train'
                    )}
                  </p>
                </div>
                
                <div className="flex gap-3">
                  <button
                    onClick={startTraining}
                    disabled={status.is_running || !datasetPath}
                    className="flex items-center gap-2 bg-green-600 hover:bg-green-500 disabled:bg-white/10 disabled:text-white/30 rounded-xl px-6 py-3 transition-all text-sm font-medium"
                  >
                    <Play className="w-4 h-4" />
                    Start
                  </button>
                  
                  <button
                    onClick={stopTraining}
                    disabled={!status.is_running}
                    className="flex items-center gap-2 bg-red-600 hover:bg-red-500 disabled:bg-white/10 disabled:text-white/30 rounded-xl px-6 py-3 transition-all text-sm font-medium"
                  >
                    <Square className="w-4 h-4" />
                    Stop
                  </button>
                </div>
              </div>

              {/* Progress Bar */}
              {status.is_running && (
                <div className="mt-6">
                  <div className="flex justify-between text-xs text-white/50 mb-2">
                    <span>Epoch {status.current_epoch}/{status.total_epochs}</span>
                    <span>{Math.round(status.progress * 100)}%</span>
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-violet-500 via-pink-500 to-orange-500 h-full rounded-full transition-all duration-500"
                      style={{ width: `${status.progress * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </SpotlightCard>

            {/* Stats Grid */}
            {status.is_running && (
              <div className="grid grid-cols-2 gap-4">
                <SpotlightCard className="p-5" spotlightColor="rgba(168, 85, 247, 0.15)">
                  <div className="text-xs text-white/40 mb-1">Current Loss</div>
                  <div className="text-3xl font-bold bg-gradient-to-r from-violet-400 to-pink-400 text-transparent bg-clip-text">
                    {status.current_loss.toFixed(4)}
                  </div>
                </SpotlightCard>
                
                <SpotlightCard className="p-5" spotlightColor="rgba(236, 72, 153, 0.15)">
                  <div className="text-xs text-white/40 mb-1">Optimizer</div>
                  <div className="text-3xl font-bold bg-gradient-to-r from-pink-400 to-orange-400 text-transparent bg-clip-text">
                    Velvet
                  </div>
                </SpotlightCard>
              </div>
            )}

            {/* Features */}
            <SpotlightCard className="p-5" spotlightColor="rgba(139, 92, 246, 0.1)">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-lg bg-violet-500/20">
                  <Sparkles className="w-5 h-5 text-violet-400" />
                </div>
                <h3 className="text-lg font-semibold">Features</h3>
              </div>
              
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                {[
                  { name: 'FlyLoRA', desc: '75% param reduction', active: true },
                  { name: 'ERA Activation', desc: 'Entropy-regulated', active: true },
                  { name: 'Velvet Optimizer', desc: 'Adaptive learning', active: true },
                  { name: 'Auto-Scaling', desc: 'Hardware optimized', active: true },
                  { name: 'Metacognition', desc: 'Self-correction', active: false },
                  { name: 'Rust/Candle', desc: 'Native performance', active: true },
                ].map((feature) => (
                  <div
                    key={feature.name}
                    className={`rounded-xl p-3 border ${
                      feature.active 
                        ? 'bg-white/5 border-white/10' 
                        : 'bg-white/[0.02] border-white/5'
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <div className={`w-1.5 h-1.5 rounded-full ${
                        feature.active ? 'bg-green-500' : 'bg-yellow-500'
                      }`}></div>
                      <span className="text-sm font-medium">{feature.name}</span>
                    </div>
                    <p className="text-xs text-white/40">{feature.desc}</p>
                  </div>
                ))}
              </div>
            </SpotlightCard>

            {/* Advanced Settings */}
            <SpotlightCard className="p-5" spotlightColor="rgba(168, 85, 247, 0.15)">
              <div 
                className="flex items-center justify-between cursor-pointer"
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-purple-500/20">
                    <Settings className="w-5 h-5 text-purple-400" />
                  </div>
                  <h3 className="text-lg font-semibold">Advanced Settings</h3>
                </div>
                <span className="text-white/40 text-sm">{showAdvanced ? '▼' : '▶'}</span>
              </div>

              {showAdvanced && (
                <div className="mt-4 space-y-4">
                  {/* Velvet Optimizer */}
                  <div className="bg-white/5 rounded-xl p-4">
                    <h4 className="text-sm font-semibold text-purple-300 mb-3">Velvet Optimizer</h4>
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-white/60">LR Multiplier</span>
                          <span className="text-white/80">{advancedConfig.velvet_lr_multiplier}x</span>
                        </div>
                        <input
                          type="range"
                          min="1"
                          max="3"
                          step="0.1"
                          value={advancedConfig.velvet_lr_multiplier}
                          onChange={(e) => setAdvancedConfig(prev => ({ ...prev, velvet_lr_multiplier: parseFloat(e.target.value) }))}
                          className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-purple-500"
                        />
                      </div>
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-white/60">Beta1 (Momentum)</span>
                          <span className="text-white/80">{advancedConfig.velvet_beta1}</span>
                        </div>
                        <input
                          type="range"
                          min="0.8"
                          max="0.99"
                          step="0.01"
                          value={advancedConfig.velvet_beta1}
                          onChange={(e) => setAdvancedConfig(prev => ({ ...prev, velvet_beta1: parseFloat(e.target.value) }))}
                          className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-purple-500"
                        />
                      </div>
                    </div>
                  </div>

                  {/* ERA Activation */}
                  <div className="bg-white/5 rounded-xl p-4">
                    <h4 className="text-sm font-semibold text-blue-300 mb-3">ERA Activation</h4>
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-white/60">Temperature</span>
                        <span className="text-white/80">{advancedConfig.era_temperature}</span>
                      </div>
                      <input
                        type="range"
                        min="0.1"
                        max="2"
                        step="0.1"
                        value={advancedConfig.era_temperature}
                        onChange={(e) => setAdvancedConfig(prev => ({ ...prev, era_temperature: parseFloat(e.target.value) }))}
                        className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-blue-500"
                      />
                    </div>
                  </div>

                  {/* FlyLoRA */}
                  <div className="bg-white/5 rounded-xl p-4">
                    <h4 className="text-sm font-semibold text-green-300 mb-3">FlyLoRA</h4>
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-white/60">Rank</span>
                          <span className="text-white/80">{advancedConfig.flylora_rank}</span>
                        </div>
                        <input
                          type="range"
                          min="4"
                          max="64"
                          step="4"
                          value={advancedConfig.flylora_rank}
                          onChange={(e) => setAdvancedConfig(prev => ({ ...prev, flylora_rank: parseInt(e.target.value) }))}
                          className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-green-500"
                        />
                      </div>
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-white/60">Sparsity</span>
                          <span className="text-white/80">{(advancedConfig.flylora_sparsity * 100).toFixed(0)}%</span>
                        </div>
                        <input
                          type="range"
                          min="0.5"
                          max="0.95"
                          step="0.05"
                          value={advancedConfig.flylora_sparsity}
                          onChange={(e) => setAdvancedConfig(prev => ({ ...prev, flylora_sparsity: parseFloat(e.target.value) }))}
                          className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-green-500"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Save Button */}
                  <button
                    onClick={() => {
                      localStorage.setItem('vesper_advanced_config', JSON.stringify(advancedConfig))
                      setShowAdvanced(false)
                    }}
                    className="w-full bg-purple-600 hover:bg-purple-500 rounded-lg px-4 py-2 text-sm font-medium transition-all"
                  >
                    Save Configuration
                  </button>
                </div>
              )}
            </SpotlightCard>

            {/* Benchmark Section */}
            <SpotlightCard className="p-5" spotlightColor="rgba(59, 130, 246, 0.15)">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-blue-500/20">
                    <FlaskConical className="w-5 h-5 text-blue-400" />
                  </div>
                  <h3 className="text-lg font-semibold">Benchmark</h3>
                </div>
                <button
                  onClick={runBenchmark}
                  disabled={!datasetPath || benchmarkRunning}
                  className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:bg-white/10 disabled:text-white/30 rounded-lg px-4 py-2 transition-all text-xs font-medium"
                >
                  <BarChart3 className="w-4 h-4" />
                  {benchmarkRunning ? 'Running...' : 'AdamW vs Velvet'}
                </button>
              </div>

              {benchmarkResults && (
                <div className="space-y-4">
                  {/* Summary */}
                  <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-4">
                    <div className="text-green-400 font-semibold mb-1">
                      🏆 {benchmarkResults.winner} Wins!
                    </div>
                    <p className="text-xs text-white/60">{benchmarkResults.summary}</p>
                  </div>

                  {/* Results Grid */}
                  <div className="grid grid-cols-2 gap-3">
                    {benchmarkResults.results.map((result) => (
                      <div
                        key={result.optimizer}
                        className={`rounded-xl p-4 border ${
                          result.optimizer === benchmarkResults.winner
                            ? 'bg-green-500/10 border-green-500/30'
                            : 'bg-white/5 border-white/10'
                        }`}
                      >
                        <div className="font-semibold mb-2">{result.optimizer}</div>
                        <div className="space-y-1 text-xs">
                          <div className="flex justify-between">
                            <span className="text-white/40">Final Loss</span>
                            <span className="text-white/80">{result.final_loss.toFixed(4)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-white/40">Best Loss</span>
                            <span className="text-white/80">{result.best_loss.toFixed(4)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-white/40">Time</span>
                            <span className="text-white/80">{(result.training_time_ms / 1000).toFixed(1)}s</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-white/40">Memory</span>
                            <span className="text-white/80">{result.memory_peak_mb.toFixed(0)} MB</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {!benchmarkResults && !benchmarkRunning && (
                <p className="text-xs text-white/40 text-center py-4">
                  Sélectionnez un dataset et cliquez sur "AdamW vs Velvet"
                </p>
              )}

              {/* Save & Export Buttons */}
              {benchmarkResults && (
                <div className="mt-4 flex gap-2">
                  <button
                    onClick={async () => {
                      try {
                        const path = await invoke<string>('save_model', {
                          format: 'safetensors',
                          modelSize: selectedModel
                        })
                        alert(`Modèle sauvegardé: ${path}`)
                      } catch (e) {
                        console.error('Save failed:', e)
                        alert(`Erreur: ${e}`)
                      }
                    }}
                    className="flex-1 flex items-center justify-center gap-2 bg-green-600 hover:bg-green-500 rounded-lg px-4 py-2 text-xs font-medium transition-all"
                  >
                    <Download className="w-4 h-4" />
                    Save Model (.safetensors)
                  </button>
                  <button
                    onClick={async () => {
                      try {
                        const path = await invoke<string>('export_onnx', {
                          modelSize: selectedModel
                        })
                        alert(`ONNX exporté: ${path}`)
                      } catch (e) {
                        console.error('Export failed:', e)
                        alert(`Erreur: ${e}`)
                      }
                    }}
                    className="flex-1 flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-500 rounded-lg px-4 py-2 text-xs font-medium transition-all"
                  >
                    <Cpu className="w-4 h-4" />
                    Export ONNX
                  </button>
                </div>
              )}
            </SpotlightCard>
          </div>
        </div>

        {/* Panneau de Logs */}
        <div className="mt-6">
          <SpotlightCard className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-4" spotlightColor="rgba(100, 100, 100, 0.15)">
            <div className="flex items-center gap-2 mb-3">
              <Database className="w-4 h-4 text-white/60" />
              <h3 className="text-sm font-semibold text-white/80">Console</h3>
              <span className="text-xs text-white/40 ml-auto">{logs.length} messages</span>
            </div>
            <div className="bg-black/60 rounded-lg p-3 max-h-[500px] overflow-y-auto font-mono text-xs space-y-1">
              {logs.length === 0 ? (
                <div className="text-white/30 text-center py-4">Aucun log pour le moment...</div>
              ) : (
                logs.map((log, i) => (
                  <div key={i} className={`flex gap-2 ${
                    log.level === 'error' ? 'text-red-400' :
                    log.level === 'success' ? 'text-green-400' :
                    log.level === 'warning' ? 'text-yellow-400' :
                    'text-white/60'
                  }`}>
                    <span className="text-white/30 shrink-0">[{log.time}]</span>
                    <span className="break-all">{log.message}</span>
                  </div>
                ))
              )}
              <div ref={logsEndRef} />
            </div>
          </SpotlightCard>
        </div>

        {/* Chat Interface Modal */}
        {showChat && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
            <div className="w-full max-w-2xl mx-4 bg-gradient-to-br from-gray-900 to-black border border-white/10 rounded-2xl shadow-2xl overflow-hidden">
              {/* Header */}
              <div className="flex items-center justify-between p-4 border-b border-white/10">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-purple-500/20">
                    <MessageSquare className="w-5 h-5 text-purple-400" />
                  </div>
                  <div>
                    <h2 className="font-semibold">VesperAI Chat</h2>
                    <p className="text-xs text-white/40">
                      {loadedModelPath ? `📦 ${loadedModelPath.split(/[/\\]/).pop()}` : 'Aucun modèle chargé'}
                    </p>
                  </div>
                </div>
                <button 
                  onClick={() => setShowChat(false)}
                  className="p-2 hover:bg-white/10 rounded-lg transition-all"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Load Model Button */}
              {!loadedModelPath && (
                <div className="p-4 border-b border-white/10 space-y-3">
                  <button
                    onClick={async () => {
                      try {
                        const selected = await open({
                          filters: [{ name: 'Model', extensions: ['onnx', 'safetensors'] }]
                        })
                        if (selected && typeof selected === 'string') {
                          setLoadedModelPath(selected)
                          setChatMessages([{ 
                            role: 'assistant', 
                            content: `✅ Modèle chargé: ${selected.split(/[/\\]/).pop()}\n\nJe suis prêt à discuter ! Que voulez-vous savoir ?` 
                          }])
                        }
                      } catch (e) {
                        console.error('Load failed:', e)
                      }
                    }}
                    className="w-full flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-500 rounded-lg px-4 py-3 font-medium transition-all"
                  >
                    <FolderOpen className="w-5 h-5" />
                    Charger un modèle local
                  </button>
                  
                  <div className="flex gap-2">
                    <button
                      onClick={async () => {
                        setChatMessages([{ 
                          role: 'assistant', 
                          content: `🦙 **TinyLlama-1.1B** (recommandé)\n\n📥 Pour télécharger:\n\`\`\`bash\npip install huggingface-hub\nhuggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ./tinyllama\n\`\`\`\n\n📁 Puis chargez le fichier \`model.safetensors\` depuis le dossier téléchargé.\n\n💡 Taille: ~2.2GB | RAM: ~4GB`
                        }])
                      }}
                      className="flex-1 flex items-center justify-center gap-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg px-3 py-2 text-sm transition-all"
                    >
                      <Download className="w-4 h-4" />
                      TinyLlama
                    </button>
                    <button
                      onClick={async () => {
                        setChatMessages([{ 
                          role: 'assistant', 
                          content: `🤖 **Phi-3-mini** (Microsoft)\n\n📥 Pour télécharger:\n\`\`\`bash\npip install huggingface-hub\nhuggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir ./phi3\n\`\`\`\n\n📁 Puis chargez le fichier \`model.safetensors\` depuis le dossier téléchargé.\n\n💡 Taille: ~7.6GB | RAM: ~8GB`
                        }])
                      }}
                      className="flex-1 flex items-center justify-center gap-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg px-3 py-2 text-sm transition-all"
                    >
                      <Download className="w-4 h-4" />
                      Phi-3
                    </button>
                  </div>
                  
                  <p className="text-xs text-white/40 text-center">
                    Ou utilisez un modèle entraîné avec VesperAI
                  </p>
                </div>
              )}

              {/* Messages */}
              <div 
                className="h-96 overflow-y-auto p-4 space-y-4"
                ref={(el) => { if (el) el.scrollTop = el.scrollHeight }}
              >
                {chatMessages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-white/30">
                    <Bot className="w-12 h-12 mb-3 opacity-50" />
                    <p>Chargez un modèle pour commencer</p>
                  </div>
                ) : (
                  chatMessages.map((msg, i) => (
                    <div 
                      key={i} 
                      className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      {msg.role === 'assistant' && (
                        <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center flex-shrink-0">
                          <Bot className="w-4 h-4 text-purple-400" />
                        </div>
                      )}
                      <div className={`max-w-[80%] rounded-2xl px-4 py-2 ${
                        msg.role === 'user' 
                          ? 'bg-blue-600 text-white' 
                          : 'bg-white/10 text-white/90'
                      }`}>
                        <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                      </div>
                      {msg.role === 'user' && (
                        <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center flex-shrink-0">
                          <User className="w-4 h-4 text-blue-400" />
                        </div>
                      )}
                    </div>
                  ))
                )}
                {generating && (
                  <div className="flex gap-3">
                    <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center">
                      <Bot className="w-4 h-4 text-purple-400 animate-pulse" />
                    </div>
                    <div className="bg-white/10 rounded-2xl px-4 py-2">
                      <div className="flex gap-1">
                        <span className="w-2 h-2 bg-white/40 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></span>
                        <span className="w-2 h-2 bg-white/40 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></span>
                        <span className="w-2 h-2 bg-white/40 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Input */}
              <div className="p-4 border-t border-white/10">
                <form 
                  onSubmit={async (e) => {
                    e.preventDefault()
                    if (!chatInput.trim() || !loadedModelPath || generating) return
                    
                    const userMsg = chatInput.trim()
                    setChatInput('')
                    setChatMessages(prev => [...prev, { role: 'user', content: userMsg }])
                    setGenerating(true)
                    
                    try {
                      const response = await invoke<string>('generate_text', {
                        modelPath: loadedModelPath,
                        prompt: userMsg,
                        maxTokens: 100
                      })
                      setChatMessages(prev => [...prev, { role: 'assistant', content: response }])
                    } catch (error) {
                      setChatMessages(prev => [...prev, { 
                        role: 'assistant', 
                        content: `❌ Erreur: ${error}\n\n(Note: La génération de texte nécessite un modèle entraîné avec un vocabulaire)` 
                      }])
                    } finally {
                      setGenerating(false)
                    }
                  }}
                  className="flex gap-2"
                >
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder={loadedModelPath ? "Écrivez votre message..." : "Chargez d'abord un modèle"}
                    disabled={!loadedModelPath || generating}
                    className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-purple-500/50 disabled:opacity-50 transition-all"
                  />
                  <button
                    type="submit"
                    disabled={!chatInput.trim() || !loadedModelPath || generating}
                    className="bg-purple-600 hover:bg-purple-500 disabled:bg-white/10 disabled:text-white/30 rounded-xl px-4 py-3 transition-all"
                  >
                    <Send className="w-5 h-5" />
                  </button>
                </form>
              </div>
            </div>
          </div>
        )}

        {/* Chat CTA Button */}
        <button
          onClick={() => setShowChat(true)}
          className="fixed bottom-6 right-6 z-40 flex items-center gap-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 rounded-full px-5 py-3 shadow-lg shadow-purple-500/25 transition-all hover:scale-105"
        >
          <MessageSquare className="w-5 h-5" />
          <span className="font-medium">Chat avec le modèle</span>
        </button>
      </div>
    </div>
  )
}

export default App
