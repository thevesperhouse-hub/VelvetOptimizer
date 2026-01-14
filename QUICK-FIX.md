# ⚡ Quick Fix - Windows SDK + erikbuehler fork

## Problème 1: `kernel32.lib` introuvable

La variable d'environnement `LIB` n'est pas configurée pour pointer vers Windows SDK.

### Solution Rapide (Session actuelle seulement)

Dans ton terminal PowerShell actuel:

```powershell
# Configurer LIB pour la session
$env:LIB = "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35222\lib\x64"

# Vérifier
echo $env:LIB
```

**Note**: Ajuste les numéros de version (10.0.22621.0 et 14.44.35222) selon ta config.

### Solution Permanente (Recommandé)

```powershell
cd F:\VelvetOptimizer
.\setup-windows-sdk.ps1
```

Le script va:
1. Détecter automatiquement tes versions SDK et MSVC
2. Ajouter les chemins à la variable LIB utilisateur
3. Redémarre ton terminal après

---

## Problème 2: Fork erikbuehler

✅ **Déjà fixé dans `Cargo.toml`**

Changé de:
```toml
candle-core = { git = "https://github.com/huggingface/candle.git" }
```

Vers:
```toml
candle-core = { git = "https://github.com/EricLBuehler/candle.git", branch = "master" }
```

---

## 🚀 Test Final

Après avoir configuré LIB:

```bash
# Nettoyer cache Cargo (important!)
cargo clean

# Recompiler avec le fork erikbuehler
cargo check --workspace --no-default-features
```

---

## 🔍 Trouver Tes Versions SDK/MSVC

Si les chemins ne matchent pas:

```powershell
# Trouver SDK version
Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\Lib" | Select Name

# Trouver MSVC version
Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC" | Select Name
```

Puis remplace dans la commande `$env:LIB`.

---

## ⚠️ Important

**Toujours utiliser Developer Command Prompt** ou avoir configuré le PATH MSVC comme on a fait avant.

Sinon tu auras à la fois besoin de:
- PATH configuré (link.exe, cl.exe)
- LIB configuré (kernel32.lib, etc.)
