param(
    [string]$Python = "..\.venv\Scripts\python.exe",
    [string]$Output = "..\output\pdf\data_centre_annual_revision.pdf"
)

$ErrorActionPreference = "Stop"
$paperDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $paperDirectory
try {
    & $Python "generate_annual_results.py"
    if ($LASTEXITCODE -ne 0) { throw "Annual result generation failed." }

    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "main_new.tex"
    if ($LASTEXITCODE -ne 0) { throw "First LaTeX pass failed." }

    & bibtex "main_new"
    if ($LASTEXITCODE -ne 0) { throw "BibTeX failed." }

    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "main_new.tex"
    if ($LASTEXITCODE -ne 0) { throw "Second LaTeX pass failed." }

    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "main_new.tex"
    if ($LASTEXITCODE -ne 0) { throw "Final LaTeX pass failed." }

    $resolvedOutput = [System.IO.Path]::GetFullPath(
        (Join-Path $paperDirectory $Output)
    )
    New-Item -ItemType Directory -Force -Path (Split-Path $resolvedOutput) |
        Out-Null
    Copy-Item -LiteralPath "main_new.pdf" -Destination $resolvedOutput -Force
    Write-Host "Wrote $resolvedOutput"
}
finally {
    Pop-Location
}
