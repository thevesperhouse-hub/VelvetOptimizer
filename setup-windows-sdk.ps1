# Script pour configurer Windows SDK LIB path
# Exécute dans le Developer Command Prompt ou après avoir configuré MSVC PATH

Write-Host "=== Configuration Windows SDK LIB Path ===" -ForegroundColor Cyan

# Détecter Windows SDK
$sdkPath = "C:\Program Files (x86)\Windows Kits\10"
if (-not (Test-Path $sdkPath)) {
    Write-Host "Windows SDK introuvable à: $sdkPath" -ForegroundColor Red
    exit 1
}

# Trouver la version du SDK installée
$sdkVersions = Get-ChildItem "$sdkPath\Lib" | Sort-Object Name -Descending
if ($sdkVersions.Count -eq 0) {
    Write-Host "Aucune version SDK trouvée dans $sdkPath\Lib" -ForegroundColor Red
    exit 1
}

$sdkVersion = $sdkVersions[0].Name
Write-Host "SDK Version détectée: $sdkVersion" -ForegroundColor Green

# Trouver MSVC version
$msvcPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
if (-not (Test-Path $msvcPath)) {
    Write-Host "MSVC introuvable à: $msvcPath" -ForegroundColor Red
    exit 1
}

$msvcVersions = Get-ChildItem $msvcPath | Sort-Object Name -Descending
if ($msvcVersions.Count -eq 0) {
    Write-Host "Aucune version MSVC trouvée" -ForegroundColor Red
    exit 1
}

$msvcVersion = $msvcVersions[0].Name
Write-Host "MSVC Version détectée: $msvcVersion" -ForegroundColor Green

# Chemins LIB à ajouter
$libPaths = @(
    "$sdkPath\Lib\$sdkVersion\um\x64",
    "$sdkPath\Lib\$sdkVersion\ucrt\x64",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\$msvcVersion\lib\x64"
)

Write-Host "`nChemins LIB à configurer:" -ForegroundColor Yellow
$validLibPaths = @()
foreach ($path in $libPaths) {
    if (Test-Path $path) {
        $validLibPaths += $path
        Write-Host "  [OK] $path" -ForegroundColor Green
    } else {
        Write-Host "  [SKIP] $path" -ForegroundColor Red
    }
}

if ($validLibPaths.Count -eq 0) {
    Write-Host "`nAucun chemin LIB valide!" -ForegroundColor Red
    exit 1
}

# Obtenir LIB actuel
$currentLib = [Environment]::GetEnvironmentVariable("LIB", "User")
if (-not $currentLib) {
    $currentLib = ""
}

# Ajouter les chemins manquants
$newLibPaths = @()
foreach ($path in $validLibPaths) {
    if ($currentLib -notlike "*$path*") {
        $newLibPaths += $path
    }
}

if ($newLibPaths.Count -eq 0) {
    Write-Host "`nTous les chemins LIB sont déjà configurés!" -ForegroundColor Green
    
    # Afficher pour session courante
    Write-Host "`nPour la session PowerShell actuelle, exécute:" -ForegroundColor Cyan
    Write-Host '$env:LIB = "' -NoNewline -ForegroundColor White
    Write-Host ($validLibPaths -join ";") -NoNewline -ForegroundColor Yellow
    Write-Host '"' -ForegroundColor White
    exit 0
}

Write-Host "`n$($newLibPaths.Count) nouveau(x) chemin(s) LIB à ajouter:" -ForegroundColor Yellow
$newLibPaths | ForEach-Object { Write-Host "  + $_" -ForegroundColor Cyan }

# Demander confirmation
$confirm = Read-Host "`nAjouter à la variable LIB utilisateur? (O/N)"
if ($confirm -ne "O" -and $confirm -ne "o") {
    Write-Host "`nPour la session PowerShell actuelle seulement, exécute:" -ForegroundColor Cyan
    Write-Host '$env:LIB = "' -NoNewline -ForegroundColor White
    Write-Host ($validLibPaths -join ";") -NoNewline -ForegroundColor Yellow
    Write-Host '"' -ForegroundColor White
    exit 0
}

# Construire nouveau LIB
$newLib = if ($currentLib) { $currentLib } else { "" }
foreach ($path in $newLibPaths) {
    if ($newLib) {
        $newLib += ";$path"
    } else {
        $newLib = $path
    }
}

[Environment]::SetEnvironmentVariable("LIB", $newLib, "User")

Write-Host "`n✅ Variable LIB mise à jour!" -ForegroundColor Green
Write-Host "`n⚠️  IMPORTANT: Redémarre ton terminal pour appliquer les changements." -ForegroundColor Yellow
Write-Host "`nOu pour la session actuelle:" -ForegroundColor Cyan
Write-Host '$env:LIB = "' -NoNewline -ForegroundColor White
Write-Host ($validLibPaths -join ";") -NoNewline -ForegroundColor Yellow
Write-Host '"' -ForegroundColor White
