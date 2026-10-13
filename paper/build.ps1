# Build the current paper only. Run from any working directory.
$ErrorActionPreference = 'Stop'
function Invoke-TexCommand {
    param([string]$Executable, [string[]]$Arguments, [string]$LogPath)
    # Windows PowerShell treats harmless MiKTeX stderr notices as errors.
    # Use the process exit code while still recording every diagnostic.
    $ErrorActionPreference = 'Continue'
    & $Executable @Arguments *> $LogPath
    $commandExit = $LASTEXITCODE
    if ($commandExit -ne 0) {
        Get-Content -LiteralPath $LogPath -Tail 25
        throw "$Executable failed with exit code $commandExit; see $LogPath"
    }
}
Push-Location $PSScriptRoot
try {
    foreach ($tool in @('pdflatex', 'bibtex')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "$tool is required; install MiKTeX or TeX Live."
        }
    }
    $buildDirectory = Join-Path $PSScriptRoot 'build'
    $pdfDirectory = Join-Path $PSScriptRoot 'pdf'
    New-Item -ItemType Directory -Force -Path $buildDirectory, $pdfDirectory | Out-Null
    foreach ($document in @('main', 'main_marked', 'supplement')) {
        Invoke-TexCommand pdflatex @('-interaction=nonstopmode', '-halt-on-error', '-file-line-error', '-output-directory=build', "$document.tex") "build/$document-pass1.txt"
        Invoke-TexCommand bibtex @("build/$document") "build/$document-bibtex.txt"
        foreach ($pass in @(2, 3, 4, 5)) {
            Invoke-TexCommand pdflatex @('-interaction=nonstopmode', '-halt-on-error', '-file-line-error', '-output-directory=build', "$document.tex") "build/$document-pass$pass.txt"
            $log = Get-Content "build/$document.log" -Raw
            if ($pass -ge 3 -and $log -notmatch 'Rerun to get cross-references right') { break }
        }
        $log = Get-Content "build/$document.log" -Raw
        if ($log -match 'undefined references|undefined on input line|multiply defined|Rerun to get cross-references right') {
            throw "$document has unresolved references; see build/$document.log"
        }
        Write-Output "$document compiled"
    }
    # Publish the PDFs only after all three documents have compiled successfully.
    foreach ($document in @('main', 'main_marked', 'supplement')) {
        Copy-Item -LiteralPath "build/$document.pdf" -Destination "$pdfDirectory/$document.pdf" -Force
    }
    Write-Output "Current PDFs: $pdfDirectory"
    # Keep failed-build diagnostics, but discard successful-build intermediates.
    $resolvedBuild = (Resolve-Path -LiteralPath $buildDirectory).Path
    if ($resolvedBuild -ne (Join-Path (Resolve-Path -LiteralPath $PSScriptRoot).Path 'build')) {
        throw 'Unexpected build directory; refusing cleanup.'
    }
    Remove-Item -LiteralPath $resolvedBuild -Recurse -Force
}
finally { Pop-Location }
