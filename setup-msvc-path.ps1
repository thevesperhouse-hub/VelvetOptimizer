# Script PowerShell pour configurer MSVC PATH globalement
# Exécuter en tant qu'Administrateur

Write-Host "=== Configuration MSVC PATH Global ===" -ForegroundColor Cyan

# Chemins MSVC pour Visual Studio 2022 Community
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community"
$msvcVersion = (Get-ChildItem "$vsPath\VC\Tools\MSVC" | Sort-Object Name -Descending | Select-Object -First 1).Name

$pathsToAdd = @(
    "$vsPath\VC\Tools\MSVC\$msvcVersion\bin\Hostx64\x64",
    "$vsPath\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin",
    "$vsPath\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja",
    "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64"
)

Write-Host "`nChemins à ajouter:" -ForegroundColor Yellow
$pathsToAdd | ForEach-Object { Write-Host "  - $_" }

# Vérifier si les chemins existent
$validPaths = @()
foreach ($path in $pathsToAdd) {
    if (Test-Path $path) {
        $validPaths += $path
        Write-Host "[OK] $path" -ForegroundColor Green
    } else {
        Write-Host "[SKIP] $path (n'existe pas)" -ForegroundColor Red
    }
}

if ($validPaths.Count -eq 0) {
    Write-Host "`nAucun chemin valide trouvé!" -ForegroundColor Red
    exit 1
}

# Obtenir le PATH actuel
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

# Ajouter les chemins manquants
$newPaths = @()
foreach ($path in $validPaths) {
    if ($currentPath -notlike "*$path*") {
        $newPaths += $path
    }
}

if ($newPaths.Count -eq 0) {
    Write-Host "`nTous les chemins sont déjà dans le PATH!" -ForegroundColor Green
    exit 0
}

Write-Host "`n$($newPaths.Count) nouveau(x) chemin(s) à ajouter:" -ForegroundColor Yellow
$newPaths | ForEach-Object { Write-Host "  + $_" -ForegroundColor Cyan }

# Demander confirmation
$confirm = Read-Host "`nAjouter ces chemins au PATH utilisateur? (O/N)"
if ($confirm -ne "O" -and $confirm -ne "o") {
    Write-Host "Annulé." -ForegroundColor Red
    exit 0
}

# Ajouter au PATH
$newPath = $currentPath
foreach ($path in $newPaths) {
    $newPath += ";$path"
}

[Environment]::SetEnvironmentVariable("Path", $newPath, "User")

Write-Host "`n✅ PATH mis à jour!" -ForegroundColor Green
Write-Host "`n⚠️  IMPORTANT: Redémarrez votre terminal/VS Code pour appliquer les changements." -ForegroundColor Yellow

Write-Host "`nPour vérifier, dans un nouveau terminal:" -ForegroundColor Cyan
Write-Host "  link.exe /?" -ForegroundColor White
Write-Host "  cl.exe" -ForegroundColor White
