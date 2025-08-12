$ver = "1.14.0"
$Url = "https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v$ver/rhubarb-$ver-win64.zip"
$Dst = "rhubarb-win64.zip"
Invoke-WebRequest -Uri $Url -OutFile $Dst
Expand-Archive -Path $Dst -DestinationPath . -Force
Remove-Item $Dst
Write-Host "Rhubarb downloaded. Point 'rhubarb_path' to the extracted 'rhubarb.exe'."
