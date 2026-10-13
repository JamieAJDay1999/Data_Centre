# Rebuild every submission file and refresh the upload folder.
# Usage (from any directory):
#   ./paper/submission_20261009/finalise.ps1
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$paper = Split-Path $here -Parent
$root = Split-Path $paper -Parent
$python = 'C:\Users\jamie\AppData\Local\Programs\Python\Python312\python.exe'
$pythonPdf = 'C:\Users\jamie\AppData\Local\Programs\Python\Python311\python.exe'  # has pypdf

Push-Location $root
try {
    & $python paper/submission_20261009/build_marked.py; if ($LASTEXITCODE) { throw 'build_marked failed' }
    & "$paper\build.ps1"
    & $python paper/submission_20261009/create_documents.py; if ($LASTEXITCODE) { throw 'create_documents failed' }
} finally { Pop-Location }

$word = New-Object -ComObject Word.Application
$word.Visible = $false; $word.DisplayAlerts = 0
try {
    foreach ($name in 'cover_letter', 'response_to_reviewers', 'highlights', 'submission_metadata') {
        $doc = $word.Documents.Open("$here\$name.docx", $false, $true)
        $doc.ExportAsFixedFormat("$here\$name.pdf", 17); $doc.Close(0)
    }
} finally { $word.Quit(); [Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null }

Push-Location $paper
try {
    & $pythonPdf submission_20261009/package_manuscript.py; if ($LASTEXITCODE) { throw 'package_manuscript failed' }
} finally { Pop-Location }

$upload = "$paper\UPLOAD_TO_APPLIED_ENERGY"
New-Item -ItemType Directory -Force $upload | Out-Null
Get-ChildItem $upload -Force | Remove-Item -Recurse -Force
# The guide requires editable source (.tex), not a PDF, as the manuscript; the
# portal builds the review PDF. Figures are supplied as separate files.
$source = "$upload\01_manuscript_LaTeX_source"
New-Item -ItemType Directory $source | Out-Null
Copy-Item "$root\.codex_tmp\submission_20261009\manuscript_sources\*" $source
$files = [ordered]@{
    '01_manuscript_LaTeX_source.zip'   = 'manuscript_sources.zip'
    '02_manuscript_marked.pdf'         = 'manuscript_marked.pdf'
    '03_cover_letter.pdf'              = 'cover_letter.pdf'
    '04_response_to_reviewers.pdf'     = 'response_to_reviewers.pdf'
    '05_highlights.docx'               = 'highlights.docx'
    '06_declaration_of_interests.docx' = 'declaration_of_interests.docx'
    '07_supplementary_information.pdf' = 'supplementary_information.pdf'
    'CHECK_ONLY_manuscript_clean.pdf'  = 'manuscript_clean.pdf'
}
foreach ($target in $files.Keys) { Copy-Item "$here\$($files[$target])" "$upload\$target" }

# No data are deposited: refuse to finish if any upload still mentions a
# repository, the old DOI placeholder or data files that are not supplied.
$pattern = '(?i)zenodo|DOIPENDING|\.csv|research archive|accompanying data'
$stale = @()
foreach ($pdf in 'CHECK_ONLY_manuscript_clean.pdf', '02_manuscript_marked.pdf', '03_cover_letter.pdf', '04_response_to_reviewers.pdf', '07_supplementary_information.pdf') {
    $text = (& pdftotext -enc UTF-8 "$upload\$pdf" - 2>$null) -join ' '
    if ($text -match $pattern) { $stale += $pdf }
}
if ((Get-Content "$source\main.tex" -Raw) -match $pattern) { $stale += 'main.tex' }
if ($stale) { throw "Data-sharing wording still present in: $($stale -join ', ')" }
Write-Output "Upload folder refreshed: $upload"
