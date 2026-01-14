# 🔧 Configuration MSVC PATH - Guide Manuel

## Méthode Automatique (Recommandé)

```powershell
# Dans PowerShell (pas besoin d'admin)
cd F:\VelvetOptimizer
.\setup-msvc-path.ps1
```

---

## Méthode Manuelle (Si script ne marche pas)

### 1. Ouvrir Variables d'Environnement

1. Appuie sur `Windows + R`
2. Tape: `sysdm.cpl`
3. Onglet **"Avancé"**
4. Clic **"Variables d'environnement"**

### 2. Modifier la Variable PATH

1. Section **"Variables utilisateur"**
2. Sélectionne **"Path"**
3. Clic **"Modifier"**

### 3. Ajouter Ces Chemins (Clic "Nouveau" pour chaque ligne)

**Trouve d'abord ta version MSVC:**
```
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\
```
Prends le dossier avec le numéro de version le plus récent (ex: `14.41.34120`)

**Ajoute ces 4 chemins:**
```
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.XX.XXXXX\bin\Hostx64\x64
C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin
C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja
C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64
```

*(Remplace `14.XX.XXXXX` par ta version)*

### 4. Valider

1. Clic **"OK"** 3 fois
2. **Ferme tous les terminaux ouverts**
3. Rouvre un nouveau terminal

### 5. Vérifier

```powershell
# Dans un nouveau PowerShell/Terminal
link.exe /?
cl.exe
```

Si tu vois de l'aide au lieu de "not found" → **C'est bon!** ✅

---

## 🚀 Après Configuration

Dans **n'importe quel terminal** (PowerShell, CMD, VS Code):

```bash
cd F:\VelvetOptimizer
cargo check --workspace --no-default-features
```

Ça devrait compiler sans erreur de linker!

---

## 🔍 Troubleshooting

### Problème: "link.exe" toujours introuvable

**Solution 1**: Redémarre **complètement** Windows (pas juste le terminal)

**Solution 2**: Vérifie que le chemin existe:
```powershell
Test-Path "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
```

**Solution 3**: Utilise le Developer Command Prompt en attendant:
- Cherche "Developer Command Prompt for VS 2022"
- C'est temporaire mais ça marche

### Problème: Je ne trouve pas la version MSVC

```powershell
# Commande pour trouver automatiquement:
Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC" | Select-Object Name
```

Prends le dossier le plus récent (plus grand numéro).

---

## ⚠️ Note

**PATH Utilisateur vs PATH Système:**
- **Utilisateur**: Seulement pour toi (recommandé)
- **Système**: Pour tous les utilisateurs (nécessite admin)

On configure **Utilisateur** pour éviter de polluer le système.
