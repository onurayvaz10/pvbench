# Egitimin durumunu gosterir:  powershell -ExecutionPolicy Bypass -File kod\egitim_durum.ps1
# S2 (ICON) kosusu icin:       ... -Ayar ayar_icon -Ana ana_icon -AyarBeklenen 1584 -AnaBeklenen 3050
param([string]$Cikti = "$env:USERPROFILE\pvbench_cikti", [int]$AyarBeklenen = 3168, [int]$AnaBeklenen = 6100,
      [string]$Ayar = "ayar", [string]$Ana = "ana")
foreach ($asama in @(@($Ayar, $AyarBeklenen), @($Ana, $AnaBeklenen))) {
    $satir = 0; $hata = 0
    Get-ChildItem "$Cikti\$($asama[0])\metrics_w*.csv" -ErrorAction SilentlyContinue | ForEach-Object {
        $icerik = Get-Content $_.FullName
        $satir += [Math]::Max(0, $icerik.Count - 1)
        $hata += ($icerik | Select-String ",error: ").Count
    }
    "{0,-9} {1,5} / {2} kosu  (%{3:N1})  hata: {4}" -f $asama[0], $satir, $asama[1], (100 * $satir / $asama[1]), $hata
}
$calisan = @(Get-Content "$Cikti\pid.txt" -ErrorAction SilentlyContinue | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }).Count
"Calisan isci: $calisan"
Get-ChildItem "$Cikti\$Ayar\log_w*.txt", "$Cikti\$Ana\log_w*.txt" -ErrorAction SilentlyContinue |
    ForEach-Object { Get-Content $_.FullName -Tail 1 } | Sort-Object | Select-Object -Last 8
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
