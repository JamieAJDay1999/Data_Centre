$pp=New-Object -ComObject PowerPoint.Application
$p=$pp.Presentations.Open((Resolve-Path 'poster_A1_data_centre_flexibility.pptx').Path,$false,$true,$false)
$out=(Resolve-Path '.').Path + '\poster_preview.png'
if(Test-Path $out){Remove-Item $out -Force}
$p.SaveAs($out,18)
$p.Close()
$pp.Quit()
Get-Item $out | Select-Object Name,Length,LastWriteTime
