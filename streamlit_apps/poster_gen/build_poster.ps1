Add-Type -AssemblyName System.Drawing
$ErrorActionPreference = "Stop"

function MmToPt([double]$mm) {
    return $mm * 72.0 / 25.4
}

function RGBVal([int]$r, [int]$g, [int]$b) {
    return [int]($r + ($g * 256) + ($b * 65536))
}

function AddPanel($slide, [double]$x, [double]$y, [double]$w, [double]$h, [int]$fillRgb, [int]$lineRgb, [double]$lineWeight = 1.4) {
    $shape = $slide.Shapes.AddShape(1, $x, $y, $w, $h) # msoShapeRectangle
    $shape.Fill.ForeColor.RGB = $fillRgb
    $shape.Line.ForeColor.RGB = $lineRgb
    $shape.Line.Weight = $lineWeight
    return $shape
}

function StyleCard(
    $shape,
    [double]$fillTransparency = 0.0,
    [bool]$shadow = $true,
    [int]$shadowRgb = 0
) {
    $shape.Fill.Transparency = $fillTransparency
    if ($shadow) {
        $shape.Shadow.Visible = -1
        if ($shadowRgb -ne 0) {
            $shape.Shadow.ForeColor.RGB = $shadowRgb
        }
        $shape.Shadow.Transparency = 0.86
        $shape.Shadow.OffsetX = 1.2
        $shape.Shadow.OffsetY = 1.2
        $shape.Shadow.Blur = 2.5
    } else {
        $shape.Shadow.Visible = 0
    }
}

function AddTextBox(
    $slide,
    [double]$x,
    [double]$y,
    [double]$w,
    [double]$h,
    [string]$text,
    [string]$fontName,
    [double]$fontSize,
    [int]$fontColorRgb,
    [bool]$bold = $false,
    [int]$align = 1,
    [double]$margin = 6.0
) {
    $shape = $slide.Shapes.AddTextbox(1, $x, $y, $w, $h) # msoTextOrientationHorizontal
    $shape.Line.Visible = 0
    $shape.Fill.Visible = 0
    $shape.TextFrame2.WordWrap = -1
    $shape.TextFrame2.AutoSize = 0
    $shape.TextFrame2.MarginLeft = $margin
    $shape.TextFrame2.MarginRight = $margin
    $shape.TextFrame2.MarginTop = $margin * 0.6
    $shape.TextFrame2.MarginBottom = $margin * 0.4
    $shape.TextFrame2.TextRange.Text = $text
    $shape.TextFrame2.TextRange.Font.Name = $fontName
    $shape.TextFrame2.TextRange.Font.Size = $fontSize
    $shape.TextFrame2.TextRange.Font.Bold = $(if ($bold) { -1 } else { 0 })
    $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $fontColorRgb
    $shape.TextFrame2.TextRange.ParagraphFormat.Alignment = $align
    return $shape
}

function EmphasizePhrase(
    $shape,
    [string]$phrase,
    [int]$accentRgb
) {
    if ([string]::IsNullOrWhiteSpace($phrase)) {
        return
    }
    $text = $shape.TextFrame2.TextRange.Text
    $startIdx = 0
    while ($true) {
        $idx = $text.IndexOf($phrase, $startIdx, [System.StringComparison]::OrdinalIgnoreCase)
        if ($idx -lt 0) {
            break
        }
        $charRange = $shape.TextFrame2.TextRange.Characters($idx + 1, $phrase.Length)
        $charRange.Font.Bold = -1
        if ($accentRgb -ne 0) {
            $charRange.Font.Fill.ForeColor.RGB = $accentRgb
        }
        $startIdx = $idx + $phrase.Length
    }
}

function AddFittedPicture($slide, [string]$path, [double]$x, [double]$y, [double]$w, [double]$h) {
    $img = [System.Drawing.Image]::FromFile($path)
    $iw = [double]$img.Width
    $ih = [double]$img.Height
    $img.Dispose()
    $scale = [Math]::Min($w / $iw, $h / $ih)
    $pw = $iw * $scale
    $ph = $ih * $scale
    $px = $x + (($w - $pw) / 2.0)
    $py = $y + (($h - $ph) / 2.0)
    return $slide.Shapes.AddPicture($path, 0, -1, $px, $py, $pw, $ph)
}

$root = (Get-Location).Path

$paths = @{
    f1 = Join-Path $root "extracted_figures\p04_img01.png"
    f2 = Join-Path $root "extracted_figures\p12_img01.png"
    f3 = Join-Path $root "extracted_figures\p12_img03.png"
    f4 = Join-Path $root "extracted_figures\p13_img01.png"
    f5u = Join-Path $root "extracted_figures\p13_img02.png"
    f5d = Join-Path $root "extracted_figures\p14_img01.png"
    qr = Join-Path $root "qr_paper.png"
}

foreach ($k in $paths.Keys) {
    if (-not (Test-Path $paths[$k])) {
        throw "Missing required asset: $($paths[$k])"
    }
}

$outPptx = Join-Path $root "poster_A1_data_centre_flexibility.pptx"
$outPdf = Join-Path $root "poster_A1_data_centre_flexibility.pdf"

$C_BG = RGBVal 241 246 250        # #F1F6FA
$C_TEXT = RGBVal 16 39 63          # #10273F
$C_TEXT_SOFT = RGBVal 44 74 106    # #2C4A6A
$C_UP = RGBVal 20 126 112          # #147E70
$C_DOWN = RGBVal 202 122 31        # #CA7A1F
$C_NEUTRAL = RGBVal 91 103 123     # #5B677B
$C_PANEL = RGBVal 255 255 255
$C_PANEL_ALT = RGBVal 248 251 253  # #F8FBFD
$C_SOFT = RGBVal 209 221 233       # #D1DDE9
$C_HEAD_BG = RGBVal 229 239 248    # #E5EFF8
$C_RULE = RGBVal 39 86 128         # #275680
$C_CALL_UP_BG = RGBVal 233 247 244 # #E9F7F4
$C_CALL_DN_BG = RGBVal 252 243 229 # #FCF3E5
$C_SHADOW = RGBVal 44 66 92
$pound = [char]163

$slideW = MmToPt 594
$slideH = MmToPt 841
$margin = MmToPt 22
$gutter = MmToPt 8
$baseline = MmToPt 6
$innerW = $slideW - (2 * $margin)
$innerH = $slideH - (2 * $margin)
$colW = ($innerW - (11 * $gutter)) / 12

function ColX([int]$c) {
    return $margin + (($c - 1) * ($colW + $gutter))
}

function SpanW([int]$span) {
    return ($span * $colW) + (($span - 1) * $gutter)
}

$topH = 0.15 * $innerH
$band2H = 0.22 * $innerH
$band3H = 0.33 * $innerH
$band4H = 0.22 * $innerH
$bottomH = 0.08 * $innerH

$topY = $margin
$band2Y = $topY + $topH
$band3Y = $band2Y + $band2H
$band4Y = $band3Y + $band3H
$bottomY = $band4Y + $band4H

$thesisText = "Core message: an integrated IT-UPS-cooling strategy can reduce operating cost and provide duration-certified flexibility to the grid."
$problemText = "Global digitalisation and AI are accelerating data-centre electricity demand while power systems integrate variable renewables and new electric loads. This increases peak stress, ramping pressure, and local network congestion. Data centres therefore have a dual role: they are part of the flexibility challenge, but they also host controllable assets that can support balancing when operated as an integrated system."
$objectiveText = "This work develops and tests a whole-facility framework with two contributions. First, it computes a least-cost day-ahead operating baseline by co-optimising IT scheduling, UPS operation, and cooling/TES dispatch. Second, it quantifies duration-certified flexibility: for any requested power deviation and start time, it computes the maximum feasible provision duration."
$integratedText = "Flexibility is represented through three coupled subsystems. IT flexibility comes from shifting delay-tolerant workload while preserving constant computational demand and respecting tranche delay limits. UPS flexibility comes from bounded charge/discharge power, state-of-charge limits, and non-simultaneous charging/discharging constraints. Cooling flexibility comes from chiller modulation, TES charge/discharge, and thermodynamic temperature dynamics for inlet air, cold aisle, racks, IT components, and hot aisle. The model uses 15-minute time steps over a 24-hour horizon plus a 3-hour extension window. A 1 MW IT facility is analysed with UPS capacity of 600 kWh and TES capacity of 1000 kWh, using day-ahead electricity prices."
$scenarioText = "Scenario 1 establishes a no-flexibility base-case cost benchmark. Scenario 2 co-optimises workload, UPS, and cooling/TES to minimise operating cost under physical constraints. Scenario 3 performs a duration-aware flexibility test on top of the Scenario 2 baseline. For each start time t0 and requested deviation DeltaP, the model searches for the maximum feasible duration tau while enforcing thermal and operational constraints and a 12-slot recovery window so post-event operation remains feasible."
$resultsTextA = "Cost optimisation reduces daily operating cost from $($pound)1,659.54 to $($pound)1,493.19, delivering $($pound)166.34 savings (10.02%) without additional hardware investment. During low-price periods, the site increases grid draw to pre-charge TES/UPS and process more IT load; during peak-price periods, it reduces grid draw through deferred IT execution, UPS discharge, and cooling reallocation."
$resultsTextB = "The flexibility heatmap reveals strong temporal structure and clear asymmetry. Upward flexibility (negative DeltaP) is strongest when IT deferral headroom is available and is reinforced by cooling reduction. Downward flexibility (positive DeltaP) is concentrated later in the day and relies more on CRAC/TES and UPS charging because advancing IT workloads is constrained without pre-conditioning. A representative contrast is -100 kW upward flexibility for 6.8 h at 00:15 versus 0.2 h at 17:30. Asset-level panels show coordinated internal counter-movements that preserve the required net response."
$implicationsText = "The framework turns abstract flexibility potential into market-relevant, duration-certified products for reserve, balancing, and price-responsive demand. Operators can optimise cost first, then stack flexibility services using residual capability. System planners gain time- and magnitude-specific estimates of dependable demand-side response from data-centre assets. Because duration is certified rather than assumed, outputs can be mapped directly to products with explicit start-time and magnitude commitments."
$limitsText = "Future work: explicit revenue stacking, pre-conditioning windows for more symmetric downward IT participation, and multi-site portfolio aggregation."

$f1Caption = "F1. Integrated architecture linking grid, UPS, IT, chiller, and TES; flexibility emerges from coordinated electrical and thermal control."
$f2Caption = "F2. Optimised demand tracks price: grid draw increases off-peak and drops during peak-price hours."
$f3Caption = "F3. Stacked dispatch shows IT, CRAC, UPS, and TES coordination delivering cost-minimising operation."
$f4Caption = "F4. Duration-certified flexibility envelope: maximum feasible duration (tau) varies strongly with start time and requested +/-DeltaP."
$f5Caption = "F5. Component-level decomposition highlights asymmetry between upward (left) and downward (right) flexibility provision."

$pp = $null
$pres = $null

try {
    $pp = New-Object -ComObject PowerPoint.Application
    $pp.Visible = -1
    $pres = $pp.Presentations.Add()
    $pres.PageSetup.SlideWidth = $slideW
    $pres.PageSetup.SlideHeight = $slideH
    $slide = $pres.Slides.Add(1, 12) # ppLayoutBlank

    $bg = AddPanel $slide 0 0 $slideW $slideH $C_BG $C_BG 0
    $bg.ZOrder(1) | Out-Null # send to back

    # Soft background motifs (purely decorative, no layout changes)
    $orbTop = $slide.Shapes.AddShape(9, -120, -84, 320, 320) # msoShapeOval
    $orbTop.Fill.ForeColor.RGB = $C_HEAD_BG
    $orbTop.Fill.Transparency = 0.74
    $orbTop.Line.Visible = 0
    $orbTop.ZOrder(1) | Out-Null

    $orbBottom = $slide.Shapes.AddShape(9, ($slideW - 210), ($slideH - 240), 320, 320)
    $orbBottom.Fill.ForeColor.RGB = $C_PANEL_ALT
    $orbBottom.Fill.Transparency = 0.7
    $orbBottom.Line.Visible = 0
    $orbBottom.ZOrder(1) | Out-Null

    # Top band
    $titleShape = AddTextBox $slide $margin ($topY + 2) $innerW 124 `
        "Characterisation and Quantification of Data Centre Flexibility for Power System Support" `
        "Montserrat" 66 $C_TEXT $true 1 2
    EmphasizePhrase $titleShape "Data Centre Flexibility" $C_RULE

    AddPanel $slide $margin ($topY + 108) $innerW 4 $C_RULE $C_RULE 0 | Out-Null

    AddTextBox $slide $margin ($topY + 116) $innerW 42 `
        "Mehmet Turker Takci, James Day, Meysam Qadrdan | Cardiff University" `
        "Source Sans 3" 25 $C_TEXT_SOFT $false 1 1 | Out-Null

    $thesisBox = AddPanel $slide $margin ($topY + 166) $innerW 64 $C_HEAD_BG $C_SOFT 1.6
    StyleCard $thesisBox 0.02 $true $C_SHADOW
    AddPanel $slide ($margin + 2) ($topY + 168) 8 60 $C_UP $C_UP 0 | Out-Null
    $thesisShape = AddTextBox $slide ($margin + 12) ($topY + 174) ($innerW - 20) 48 $thesisText "Source Sans 3" 24 $C_TEXT $true 1 2
    EmphasizePhrase $thesisShape "reduce operating cost" $C_UP
    EmphasizePhrase $thesisShape "duration-certified flexibility" $C_DOWN

    # Band 2
    $f1PanelY = $band2Y + 16
    $f1PanelH = $band2H - 54
    $f1Panel = AddPanel $slide (ColX 1) $f1PanelY (SpanW 6) $f1PanelH $C_PANEL_ALT $C_SOFT 1.5
    StyleCard $f1Panel 0.0 $true $C_SHADOW
    AddFittedPicture $slide $paths.f1 ((ColX 1) + 10) ($f1PanelY + 10) ((SpanW 6) - 20) ($f1PanelH - 54) | Out-Null
    $f1Cap = AddTextBox $slide (ColX 1) ($f1PanelY + $f1PanelH - 40) (SpanW 6) 36 $f1Caption "Source Sans 3" 18 $C_NEUTRAL $false 1 1
    $f1Cap.TextFrame2.TextRange.Font.Italic = -1

    $textX = ColX 7
    $textW = SpanW 6
    AddPanel $slide ($textX - 6) ($band2Y + 12) 4 26 $C_RULE $C_RULE 0 | Out-Null
    AddTextBox $slide $textX ($band2Y + 10) $textW 32 "Problem and Motivation" "Montserrat" 25 $C_TEXT $true 1 1 | Out-Null
    $problemShape = AddTextBox $slide $textX ($band2Y + 38) $textW 116 $problemText "Source Sans 3" 18.4 $C_TEXT $false 1 1
    EmphasizePhrase $problemShape "dual role" $C_RULE
    EmphasizePhrase $problemShape "integrated system" $C_UP

    AddPanel $slide ($textX - 6) ($band2Y + 156) 4 26 $C_RULE $C_RULE 0 | Out-Null
    AddTextBox $slide $textX ($band2Y + 154) $textW 32 "Research Objective" "Montserrat" 25 $C_TEXT $true 1 1 | Out-Null
    $objectiveShape = AddTextBox $slide $textX ($band2Y + 182) $textW 110 $objectiveText "Source Sans 3" 18.3 $C_TEXT $false 1 1
    EmphasizePhrase $objectiveShape "least-cost day-ahead operating baseline" $C_UP
    EmphasizePhrase $objectiveShape "duration-certified flexibility" $C_DOWN

    AddPanel $slide ($textX - 6) ($band2Y + 294) 4 26 $C_RULE $C_RULE 0 | Out-Null
    AddTextBox $slide $textX ($band2Y + 292) $textW 32 "Integrated Model" "Montserrat" 25 $C_TEXT $true 1 1 | Out-Null
    $integratedShape = AddTextBox $slide $textX ($band2Y + 320) $textW ($band2H - 324) $integratedText "Source Sans 3" 17.2 $C_TEXT $false 1 1
    EmphasizePhrase $integratedShape "three coupled subsystems" $C_RULE
    EmphasizePhrase $integratedShape "1 MW IT facility" $C_UP

    # Band 3 (F2 + F3 + scenario/results)
    $leftX = ColX 1
    $rightX = ColX 7
    $halfW = SpanW 6
    $imgTop = $band3Y + 12
    $imgH = 380

    $f2Panel = AddPanel $slide $leftX $imgTop $halfW $imgH $C_PANEL_ALT $C_SOFT 1.5
    StyleCard $f2Panel 0.0 $true $C_SHADOW
    AddFittedPicture $slide $paths.f2 ($leftX + 10) ($imgTop + 10) ($halfW - 20) ($imgH - 50) | Out-Null
    $f2Cap = AddTextBox $slide $leftX ($imgTop + $imgH - 38) $halfW 34 $f2Caption "Source Sans 3" 18 $C_NEUTRAL $false 1 1
    $f2Cap.TextFrame2.TextRange.Font.Italic = -1

    $f3Panel = AddPanel $slide $rightX $imgTop $halfW $imgH $C_PANEL_ALT $C_SOFT 1.5
    StyleCard $f3Panel 0.0 $true $C_SHADOW
    AddFittedPicture $slide $paths.f3 ($rightX + 10) ($imgTop + 10) ($halfW - 20) ($imgH - 50) | Out-Null
    $f3Cap = AddTextBox $slide $rightX ($imgTop + $imgH - 38) $halfW 34 $f3Caption "Source Sans 3" 18 $C_NEUTRAL $false 1 1
    $f3Cap.TextFrame2.TextRange.Font.Italic = -1

    AddPanel $slide ($leftX - 6) ($imgTop + $imgH + 14) 4 32 $C_UP $C_UP 0 | Out-Null
    AddTextBox $slide $leftX ($imgTop + $imgH + 10) $halfW 42 "Scenario Design" "Montserrat" 32 $C_TEXT $true 1 1 | Out-Null
    $scenarioShape = AddTextBox $slide $leftX ($imgTop + $imgH + 50) $halfW 250 $scenarioText "Source Sans 3" 21 $C_TEXT $false 1 1
    EmphasizePhrase $scenarioShape "Scenario 3" $C_DOWN
    EmphasizePhrase $scenarioShape "maximum feasible duration tau" $C_RULE

    AddPanel $slide ($rightX - 6) ($imgTop + $imgH + 14) 4 32 $C_UP $C_UP 0 | Out-Null
    AddTextBox $slide $rightX ($imgTop + $imgH + 10) $halfW 42 "Key Results (Cost and Dispatch)" "Montserrat" 32 $C_TEXT $true 1 1 | Out-Null
    $resultsAShape = AddTextBox $slide $rightX ($imgTop + $imgH + 50) $halfW 250 $resultsTextA "Source Sans 3" 21 $C_TEXT $false 1 1
    EmphasizePhrase $resultsAShape "$($pound)166.34 savings" $C_UP
    EmphasizePhrase $resultsAShape "10.02%" $C_UP

    # Mandatory callout near F2
    $callout1 = AddPanel $slide ($leftX + 16) ($imgTop + 18) 290 108 $C_CALL_UP_BG $C_UP 2.4
    StyleCard $callout1 0.0 $true $C_SHADOW
    AddTextBox $slide ($leftX + 24) ($imgTop + 26) 274 92 "$($pound)1,659.54 -> $($pound)1,493.19`nSavings: $($pound)166.34 (10.02%)" "IBM Plex Mono" 18.5 $C_TEXT $true 1 1 | Out-Null

    # Band 4 (F4 focal visual + callouts + results)
    $f4PanelY = $band4Y + 12
    $f4PanelH = $band4H - 20
    $f4Panel = AddPanel $slide (ColX 1) $f4PanelY $innerW $f4PanelH $C_PANEL_ALT $C_SOFT 1.7
    StyleCard $f4Panel 0.0 $true $C_SHADOW
    AddFittedPicture $slide $paths.f4 ((ColX 1) + 10) ($f4PanelY + 12) ($innerW - 20) ($f4PanelH - 88) | Out-Null
    $f4Cap = AddTextBox $slide (ColX 1) ($f4PanelY + $f4PanelH - 72) $innerW 34 $f4Caption "Source Sans 3" 18 $C_NEUTRAL $false 1 1
    $f4Cap.TextFrame2.TextRange.Font.Italic = -1

    # Mandatory callouts near F4
    $callout2 = AddPanel $slide ((ColX 9) + 10) ($f4PanelY + 24) (SpanW 4 - 20) 82 $C_CALL_DN_BG $C_DOWN 2.4
    StyleCard $callout2 0.0 $true $C_SHADOW
    AddTextBox $slide ((ColX 9) + 18) ($f4PanelY + 30) (SpanW 4 - 36) 68 "-100 kW upward flexibility: 6.8 h at 00:15`nvs 0.2 h at 17:30" "IBM Plex Mono" 15.4 $C_TEXT $true 1 1 | Out-Null

    $callout3 = AddPanel $slide ((ColX 9) + 10) ($f4PanelY + 114) (SpanW 4 - 20) 70 $C_CALL_UP_BG $C_UP 2.4
    StyleCard $callout3 0.0 $true $C_SHADOW
    AddTextBox $slide ((ColX 9) + 18) ($f4PanelY + 120) (SpanW 4 - 36) 58 "Model scope: 1 MW IT | UPS 600 kWh | TES 1000 kWh" "IBM Plex Mono" 14.2 $C_TEXT $true 1 1 | Out-Null

    AddPanel $slide ((ColX 1) - 6) ($f4PanelY + $f4PanelH - 38) 4 30 $C_DOWN $C_DOWN 0 | Out-Null
    AddTextBox $slide (ColX 1) ($f4PanelY + $f4PanelH - 40) (SpanW 8) 50 "Key Results (Flexibility Asymmetry)" "Montserrat" 30 $C_TEXT $true 1 1 | Out-Null
    $resultsBShape = AddTextBox $slide (ColX 1) ($f4PanelY + $f4PanelH + 6) $innerW 64 $resultsTextB "Source Sans 3" 19 $C_TEXT $false 1 1
    EmphasizePhrase $resultsBShape "clear asymmetry" $C_DOWN
    EmphasizePhrase $resultsBShape "6.8 h at 00:15 versus 0.2 h at 17:30" $C_DOWN

    # Bottom band: F5 pair + implications + future work + QR/reference strip
    $bottomPad = 6
    $imgBottomH = $bottomH - 24

    $f5LeftPanel = AddPanel $slide (ColX 1) ($bottomY + $bottomPad) (SpanW 4) $imgBottomH $C_PANEL_ALT $C_SOFT 1.4
    StyleCard $f5LeftPanel 0.0 $true $C_SHADOW
    AddFittedPicture $slide $paths.f5u ((ColX 1) + 8) ($bottomY + $bottomPad + 8) ((SpanW 4) - 16) ($imgBottomH - 16) | Out-Null

    $f5RightPanel = AddPanel $slide (ColX 5) ($bottomY + $bottomPad) (SpanW 4) $imgBottomH $C_PANEL_ALT $C_SOFT 1.4
    StyleCard $f5RightPanel 0.0 $true $C_SHADOW
    AddFittedPicture $slide $paths.f5d ((ColX 5) + 8) ($bottomY + $bottomPad + 8) ((SpanW 4) - 16) ($imgBottomH - 16) | Out-Null

    AddTextBox $slide (ColX 1) ($bottomY - 18) (SpanW 8) 22 $f5Caption "Source Sans 3" 16 $C_NEUTRAL $false 1 1 | Out-Null

    $rightPanel = AddPanel $slide (ColX 9) ($bottomY + $bottomPad) (SpanW 4) $imgBottomH $C_PANEL_ALT $C_SOFT 1.4
    StyleCard $rightPanel 0.0 $true $C_SHADOW
    AddPanel $slide ((ColX 9) + 4) ($bottomY + 12) 4 24 $C_RULE $C_RULE 0 | Out-Null
    AddTextBox $slide ((ColX 9) + 8) ($bottomY + 10) ((SpanW 4) - 16) 28 "Practical Implications" "Montserrat" 20 $C_TEXT $true 1 1 | Out-Null
    $implicationsShape = AddTextBox $slide ((ColX 9) + 8) ($bottomY + 34) ((SpanW 4) - 16) 72 $implicationsText "Source Sans 3" 14 $C_TEXT $false 1 1
    EmphasizePhrase $implicationsShape "duration-certified products" $C_UP
    EmphasizePhrase $implicationsShape "stack flexibility services" $C_RULE
    AddTextBox $slide ((ColX 9) + 8) ($bottomY + 108) ((SpanW 4) - 16) 24 "Limitations and Next Work" "Montserrat" 16 $C_TEXT $true 1 1 | Out-Null
    $limitsShape = AddTextBox $slide ((ColX 9) + 8) ($bottomY + 126) ((SpanW 4) - 118) 40 $limitsText "Source Sans 3" 13.2 $C_TEXT $false 1 1
    EmphasizePhrase $limitsShape "multi-site portfolio aggregation" $C_DOWN

    # QR and references strip
    AddFittedPicture $slide $paths.qr ((ColX 12) - 74) ($bottomY + 120) 66 66 | Out-Null
    AddTextBox $slide ((ColX 9) + 8) ($bottomY + 160) ((SpanW 4) - 84) 18 "QR: full paper PDF" "IBM Plex Mono" 10 $C_NEUTRAL $false 1 1 | Out-Null
    AddTextBox $slide $margin ($slideH - 20) $innerW 16 "References: IEA (2025) Energy and AI; ASHRAE thermal guidance; Takci et al. (2025)." "Source Sans 3" 10 $C_NEUTRAL $false 1 1 | Out-Null

    if (Test-Path $outPptx) { Remove-Item $outPptx -Force }
    if (Test-Path $outPdf) { Remove-Item $outPdf -Force }

    $pres.SaveAs($outPptx)
    $pres.SaveAs($outPdf, 32) # ppSaveAsPDF
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

Write-Output "Generated:"
Write-Output $outPptx
Write-Output $outPdf


