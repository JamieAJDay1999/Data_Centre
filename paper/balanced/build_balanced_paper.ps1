param(
    [string]$Output = "..\..\output\pdf\data_centre_balanced_revision.pdf"
)

$ErrorActionPreference = "Stop"
$draftDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$paperDirectory = Split-Path -Parent $draftDirectory
Push-Location $paperDirectory
try {
    $oldTexInputs = $env:TEXINPUTS
    $oldBibInputs = $env:BIBINPUTS
    $env:TEXINPUTS = ".;balanced;" + $oldTexInputs
    $env:BIBINPUTS = ".;balanced;" + $oldBibInputs

    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=balanced "balanced/main_balanced.tex"
    if ($LASTEXITCODE -ne 0) { throw "First LaTeX pass failed." }
    Start-Sleep -Milliseconds 750

    & bibtex "balanced/main_balanced"
    if ($LASTEXITCODE -ne 0) { throw "BibTeX failed." }

    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=balanced "balanced/main_balanced.tex"
    if ($LASTEXITCODE -ne 0) { throw "Second LaTeX pass failed." }
    Start-Sleep -Milliseconds 750

    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=balanced "balanced/main_balanced.tex"
    if ($LASTEXITCODE -ne 0) { throw "Final LaTeX pass failed." }

    $resolvedOutput = [System.IO.Path]::GetFullPath(
        (Join-Path $draftDirectory $Output)
    )
    New-Item -ItemType Directory -Force -Path (Split-Path $resolvedOutput) |
        Out-Null
    Copy-Item -LiteralPath (Join-Path $draftDirectory "main_balanced.pdf") -Destination $resolvedOutput -Force
    Write-Host "Wrote $resolvedOutput"
}
finally {
    $env:TEXINPUTS = $oldTexInputs
    $env:BIBINPUTS = $oldBibInputs
    Pop-Location
}
