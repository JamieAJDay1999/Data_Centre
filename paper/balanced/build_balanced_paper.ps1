param(
    [string]$Output = "..\..\output\pdf\data_centre_balanced_revision.pdf"
)

$ErrorActionPreference = "Stop"
$draftDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $draftDirectory
try {
    $oldTexInputs = $env:TEXINPUTS
    $oldBibInputs = $env:BIBINPUTS
    $env:TEXINPUTS = ".;..;" + $oldTexInputs
    $env:BIBINPUTS = ".;..;" + $oldBibInputs

    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "main_balanced.tex"
    if ($LASTEXITCODE -ne 0) { throw "First LaTeX pass failed." }

    & bibtex "main_balanced"
    if ($LASTEXITCODE -ne 0) { throw "BibTeX failed." }

    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "main_balanced.tex"
    if ($LASTEXITCODE -ne 0) { throw "Second LaTeX pass failed." }

    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "main_balanced.tex"
    if ($LASTEXITCODE -ne 0) { throw "Final LaTeX pass failed." }

    $resolvedOutput = [System.IO.Path]::GetFullPath(
        (Join-Path $draftDirectory $Output)
    )
    New-Item -ItemType Directory -Force -Path (Split-Path $resolvedOutput) |
        Out-Null
    Copy-Item -LiteralPath "main_balanced.pdf" -Destination $resolvedOutput -Force
    Write-Host "Wrote $resolvedOutput"
}
finally {
    $env:TEXINPUTS = $oldTexInputs
    $env:BIBINPUTS = $oldBibInputs
    Pop-Location
}
