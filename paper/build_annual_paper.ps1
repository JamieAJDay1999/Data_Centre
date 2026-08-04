param(
    [string]$Python = "..\.venv\Scripts\python.exe",
    [string]$Output = "..\output\pdf\data_centre_annual_revision.pdf"
)

$ErrorActionPreference = "Stop"
$paperDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $paperDirectory
try {
    # Resolved here because the flexibility figures are generated from the
    # repository root, where a paper-relative interpreter path would not exist.
    $Python = [System.IO.Path]::GetFullPath((Join-Path $paperDirectory $Python))

    & $Python "generate_annual_results.py"
    if ($LASTEXITCODE -ne 0) { throw "Annual result generation failed." }

    # Redraws Figures 6-8 from the cached flexibility sweep; no MILP is solved.
    Push-Location ".."
    try {
        & $Python -m rolling_optimisation.plot_flexibility_figures
        if ($LASTEXITCODE -ne 0) { throw "Flexibility figure generation failed." }
    }
    finally { Pop-Location }

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
