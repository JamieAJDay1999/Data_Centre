Add-Type -AssemblyName System.Drawing
$ErrorActionPreference = "Stop"

function RGBVal([int]$r, [int]$g, [int]$b) {
    return [int]($r + ($g * 256) + ($b * 65536))
}

function IsHeadingText([string]$text) {
    if ([string]::IsNullOrWhiteSpace($text)) { return $false }
    $t = $text.Trim()
    return $t -match "^(Problem and Motivation|Research Objective|Integrated Model|Scenario Design|Key Results|Practical Implications|Limitations and Next Work)"
}

function IsFigureCaption([string]$text) {
    if ([string]::IsNullOrWhiteSpace($text)) { return $false }
    return $text.Trim() -match "^F\d+\."
}

$root = (Get-Location).Path
$pptxPath = Join-Path $root "poster_A1_data_centre_flexibility.pptx"

if (-not (Test-Path $pptxPath)) {
    throw "Missing poster file: $pptxPath"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupPath = Join-Path $root ("poster_A1_data_centre_flexibility_before_polish_" + $timestamp + ".pptx")
Copy-Item $pptxPath $backupPath -Force

$C_BG = RGBVal 246 249 252
$C_TEXT = RGBVal 23 37 55
$C_TEXT_SOFT = RGBVal 59 78 100
$C_ACCENT = RGBVal 25 94 138
$C_PANEL = RGBVal 255 255 255
$C_PANEL_ALT = RGBVal 250 252 255
$C_BORDER = RGBVal 206 220 234
$C_CAPTION = RGBVal 87 103 124
$C_CALLOUT = RGBVal 234 244 250
$C_IMG_BORDER = RGBVal 180 201 220

$pp = $null
$pres = $null

try {
    $pp = New-Object -ComObject PowerPoint.Application
    $pp.Visible = -1
    $pres = $pp.Presentations.Open($pptxPath, $false, $false, $false)

    for ($si = 1; $si -le $pres.Slides.Count; $si++) {
        $slide = $pres.Slides.Item($si)
        $slideW = [double]$pres.PageSetup.SlideWidth
        $slideH = [double]$pres.PageSetup.SlideHeight

        # Normalize background for cleaner contrast.
        $slide.FollowMasterBackground = 0
        $slide.Background.Fill.Visible = -1
        $slide.Background.Fill.Solid()
        $slide.Background.Fill.ForeColor.RGB = $C_BG

        for ($i = 1; $i -le $slide.Shapes.Count; $i++) {
            $shape = $slide.Shapes.Item($i)

            if ($shape.Type -eq 1) { # msoAutoShape
                $w = [double]$shape.Width
                $h = [double]$shape.Height
                $top = [double]$shape.Top

                if ($w -gt ($slideW * 0.95) -and $h -gt ($slideH * 0.95)) {
                    $shape.Fill.ForeColor.RGB = $C_BG
                    $shape.Line.Visible = 0
                } elseif (($w -le 10 -and $h -gt 20) -or ($h -le 10 -and $w -gt 120)) {
                    # Keep dividers/section markers crisp.
                    $shape.Fill.ForeColor.RGB = $C_ACCENT
                    $shape.Line.Visible = 0
                } elseif ($w -gt 180 -and $h -gt 70) {
                    # Card/panel treatment.
                    if ($w -lt 360 -and $h -lt 140) {
                        $shape.Fill.ForeColor.RGB = $C_CALLOUT
                    } elseif ($top -lt ($slideH * 0.18)) {
                        $shape.Fill.ForeColor.RGB = $C_PANEL_ALT
                    } else {
                        $shape.Fill.ForeColor.RGB = $C_PANEL
                    }
                    $shape.Fill.Transparency = 0
                    $shape.Line.Visible = -1
                    $shape.Line.ForeColor.RGB = $C_BORDER
                    $shape.Line.Weight = 1.2
                    $shape.Shadow.Visible = -1
                    $shape.Shadow.Transparency = 0.9
                    $shape.Shadow.OffsetX = 1.2
                    $shape.Shadow.OffsetY = 1.2
                    $shape.Shadow.Blur = 2.4
                }
            }

            if ($shape.Type -eq 13) { # msoPicture
                $shape.Line.Visible = -1
                $shape.Line.ForeColor.RGB = $C_IMG_BORDER
                $shape.Line.Weight = 1.0
                $shape.Shadow.Visible = -1
                $shape.Shadow.Transparency = 0.9
                $shape.Shadow.OffsetX = 1
                $shape.Shadow.OffsetY = 1
                $shape.Shadow.Blur = 2
            }

            if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
                $text = [string]$shape.TextFrame.TextRange.Text
                if ([string]::IsNullOrWhiteSpace($text)) { continue }

                $size = 18.0
                try { $size = [double]$shape.TextFrame.TextRange.Font.Size } catch {}

                # Improve spacing consistency while preserving shape positions.
                $shape.TextFrame2.MarginLeft = 5
                $shape.TextFrame2.MarginRight = 5
                $shape.TextFrame2.MarginTop = 3
                $shape.TextFrame2.MarginBottom = 2
                $shape.TextFrame2.WordWrap = -1

                $pf = $shape.TextFrame2.TextRange.ParagraphFormat
                $pf.SpaceWithin = 1.1
                $pf.SpaceAfter = 1.5

                if ($size -ge 56) {
                    $shape.TextFrame2.TextRange.Font.Name = "Segoe UI Semibold"
                    $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $C_TEXT
                    $shape.TextFrame2.TextRange.Font.Bold = -1
                } elseif ($text.Trim().StartsWith("Core message:", [System.StringComparison]::OrdinalIgnoreCase)) {
                    $shape.TextFrame2.TextRange.Font.Name = "Segoe UI Semibold"
                    $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $C_TEXT_SOFT
                    $shape.TextFrame2.TextRange.Font.Bold = -1
                } elseif (IsHeadingText $text -or $size -ge 28) {
                    $shape.TextFrame2.TextRange.Font.Name = "Segoe UI Semibold"
                    $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $C_TEXT
                    $shape.TextFrame2.TextRange.Font.Bold = -1
                } elseif (IsFigureCaption $text) {
                    $shape.TextFrame2.TextRange.Font.Name = "Segoe UI"
                    $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $C_CAPTION
                    $shape.TextFrame2.TextRange.Font.Italic = -1
                    $shape.TextFrame2.TextRange.Font.Bold = 0
                } elseif ($size -le 11) {
                    $shape.TextFrame2.TextRange.Font.Name = "Segoe UI"
                    $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $C_CAPTION
                    $shape.TextFrame2.TextRange.Font.Bold = 0
                } elseif ($text -match "Savings:|->|Model scope:") {
                    $shape.TextFrame2.TextRange.Font.Name = "Segoe UI Semibold"
                    $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $C_TEXT
                    $shape.TextFrame2.TextRange.Font.Bold = -1
                } else {
                    $shape.TextFrame2.TextRange.Font.Name = "Segoe UI"
                    $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $C_TEXT_SOFT
                    $shape.TextFrame2.TextRange.Font.Bold = 0
                }
            }
        }
    }

    $pres.Save()
}
finally {
    if ($pres) {
        $pres.Close()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres)
    }
    if ($pp) {
        $pp.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($pp)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Write-Output "Polished poster in place:"
Write-Output $pptxPath
Write-Output "Backup created:"
Write-Output $backupPath
