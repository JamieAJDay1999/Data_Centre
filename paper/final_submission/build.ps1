$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force -Path build | Out-Null
    foreach ($document in @('main','supplement')) {
        & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build "$document.tex" *> "build/$document-pass1.txt"
        if ($LASTEXITCODE -ne 0) { Get-Content "build/$document-pass1.txt" -Tail 30; throw "$document first pass failed" }
        & bibtex "build/$document" *> "build/$document-bibtex.txt"
        if ($LASTEXITCODE -ne 0) { Get-Content "build/$document-bibtex.txt" -Tail 30; throw "$document bibliography failed" }
        foreach ($pass in @(2,3)) {
            & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build "$document.tex" *> "build/$document-pass$pass.txt"
            if ($LASTEXITCODE -ne 0) { Get-Content "build/$document-pass$pass.txt" -Tail 30; throw "$document pass $pass failed" }
        }
        Write-Output "$document compiled"
    }
    $outputDirectory = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'output/pdf'
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
    Copy-Item -LiteralPath build/main.pdf -Destination (Join-Path $outputDirectory 'data_centre_final_submission.pdf') -Force
    Copy-Item -LiteralPath build/supplement.pdf -Destination (Join-Path $outputDirectory 'data_centre_final_supplement.pdf') -Force
}
finally { Pop-Location }
