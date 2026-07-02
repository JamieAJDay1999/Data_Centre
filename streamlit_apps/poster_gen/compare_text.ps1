$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
$before = Join-Path $root 'poster_A1_data_centre_flexibility_before_polish_20260216_121014.pptx'
$after = Join-Path $root 'poster_A1_data_centre_flexibility.pptx'
$pp = New-Object -ComObject PowerPoint.Application
$pp.Visible = -1

function Get-TextDump($pp, [string]$path){
  $pres = $pp.Presentations.Open($path,$false,$true,$false)
  $lines = New-Object System.Collections.Generic.List[string]
  for($si=1;$si -le $pres.Slides.Count;$si++){
    $slide = $pres.Slides.Item($si)
    for($i=1;$i -le $slide.Shapes.Count;$i++){
      $s=$slide.Shapes.Item($i)
      try {
        if($s.HasTextFrame -eq -1 -and $s.TextFrame.HasText -eq -1){
          $t = ($s.TextFrame.TextRange.Text -replace "`r"," " -replace "`n"," ").Trim()
          if($t.Length -gt 0){ [void]$lines.Add($t) }
        }
      } catch {}
    }
  }
  $pres.Close()
  [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres)
  return $lines
}

$tb = Get-TextDump $pp $before
$ta = Get-TextDump $pp $after
$pp.Quit()
[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($pp)

Write-Output "before_count=$($tb.Count) after_count=$($ta.Count)"
$diff = Compare-Object -ReferenceObject $tb -DifferenceObject $ta -SyncWindow 0
if($null -eq $diff){
  Write-Output 'TEXT_IDENTICAL=True'
}else{
  Write-Output 'TEXT_IDENTICAL=False'
  $diff | Select-Object -First 20 | ForEach-Object { Write-Output ("[$($_.SideIndicator)] $($_.InputObject)") }
}
