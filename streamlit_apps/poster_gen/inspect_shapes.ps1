$ErrorActionPreference = 'Stop'
$pptx = Join-Path (Get-Location) 'poster_A1_data_centre_flexibility.pptx'
$pp = New-Object -ComObject PowerPoint.Application
$pp.Visible = -1
$pres = $pp.Presentations.Open($pptx,$false,$true,$false)
$slide = $pres.Slides.Item(1)
"Slide: $($slide.SlideNumber) size=[$($pres.PageSetup.SlideWidth)x$($pres.PageSetup.SlideHeight)] shapes=$($slide.Shapes.Count)"
for($i=1;$i -le $slide.Shapes.Count;$i++){
  $s = $slide.Shapes.Item($i)
  $txt=''
  $hasText=$false
  try { if($s.HasTextFrame -eq -1 -and $s.TextFrame.HasText -eq -1){ $hasText=$true; $txt = $s.TextFrame.TextRange.Text } } catch {}
  if($txt.Length -gt 100){ $txt = $txt.Substring(0,100).Replace("`r"," ").Replace("`n"," ") + '...' }
  $size = ''
  try { if($hasText){ $size = [double]$s.TextFrame.TextRange.Font.Size } } catch {}
  "#{0,2} type={1,2} left={2,7:N1} top={3,7:N1} w={4,7:N1} h={5,7:N1} text={6} size={7} '{8}'" -f $i,$s.Type,$s.Left,$s.Top,$s.Width,$s.Height,$hasText,$size,$txt
}
$pres.Close()
$pp.Quit()
[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres)
[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($pp)
