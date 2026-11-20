# E8: cozunurluk duyarliligi (pvbench.resolution): 5, 10 ve 20 dakikada bir adim ileri tahmin, S0 girdileri.
# Once tablolar kurulur:  python -m pvbench.resolution --build   (veri\islenmis\altsaatlik, altsaatlik_envanter.csv)
#   Sinir aglari (GPU, P cekirdekleri), onceki zincir bitince:
#       egitim_e8.ps1 -Ad e8 -Model "gru tcn" -Tohum 3 -OncekiPid e6_zincir_pid.txt
#   Referanslar (yalniz islemci; verim cekirdeklerinde, GPU kuyruguyla ayni anda):
#       egitim_e8.ps1 -Ad e8_ref -Model "persistence smart_persistence ridge lightgbm" -Verim -OncekiPid yok
# Sonuc <Cikti>\<Ad>\metrics_w*.csv (uygun santral x 3 cozunurluk x model x tohum satir).  Gunluk: <Cikti>\<Ad>_zincir.txt
# Baslatma: egitim_e2.ps1 ile ayni (WMI).  -Verim kosusu pid.txt'ye dokunmaz (<Ad>_pid.txt kullanir).
param([string]$Cikti = "$env:USERPROFILE\pvbench_cikti", [string]$Ad = "e8", [int]$Isci = 8, [int]$Tohum = 3,
      [string]$Model = "gru tcn", [string]$Ayar = "ayar_e2", [string]$OncekiPid = "e6_zincir_pid.txt", [switch]$Verim)
$py = "$env:USERPROFILE\.venvs\pvbench\Scripts\python.exe"
$kod = $PSScriptRoot
$gunluk = "$Cikti\${Ad}_zincir.txt"
$pidDosya = if ($Verim) { "$Cikti\${Ad}_pid.txt" } else { "$Cikti\pid.txt" }
function Yaz($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Add-Content $gunluk -Encoding utf8 }
function Satir($desen) {
    $n = 0
    Get-ChildItem $desen -ErrorAction SilentlyContinue | ForEach-Object { $n += @(Import-Csv $_.FullName).Count }
    $n
}
function Bekle($dosya) {
    while (@(Get-Content $dosya -ErrorAction SilentlyContinue | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }).Count -gt 0) {
        Start-Sleep -Seconds 60
    }
}
Add-Type -Namespace Win -Name Guc -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Win.Guc]::SetThreadExecutionState(0x80000001) | Out-Null

Yaz "$Ad zinciri basladi; modeller [$Model], onceki zincir [$OncekiPid]"
$onceki = Get-Content "$Cikti\$OncekiPid" -ErrorAction SilentlyContinue
while ($onceki -and (Get-Process -Id $onceki -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 60 }
if (-not $Verim) { Bekle $pidDosya }

$modeller = @($Model -split "\s+" | Where-Object { $_ })
$envanter = Import-Csv "$kod\..\veri\islenmis\altsaatlik_envanter.csv" -Delimiter ";" -Encoding UTF8
$nSantral = @($envanter | Where-Object { $_.eligible -eq "True" }).Count
$nTohum = if ($Verim) { 1 } else { $Tohum }
$beklenen = $nSantral * 3 * $modeller.Count * $nTohum
$gizli = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
New-Item -ItemType Directory -Force "$Cikti\$Ad" | Out-Null
foreach ($deneme in 1..3) {
    Yaz "$Ad basliyor (deneme $deneme): $nSantral santral, beklenen $beklenen satir"
    Remove-Item "$Cikti\$Ad\claims" -Recurse -Force -ErrorAction SilentlyContinue
    $pids = @()
    for ($i = 0; $i -lt $Isci; $i++) {
        $betik = "$Cikti\${Ad}_$i.cmd"
        $satirlar = @("@echo off", "set PYTHONIOENCODING=utf-8")
        if ($Verim) { $satirlar += "set PVBENCH_AFFINITY=$('{0:X}' -f ([int64]3 -shl (16 + 2 * $i)))" }
        $satirlar += @("cd /d `"$kod`"",
          "`"$py`" -W ignore -m pvbench.resolution --out `"$Cikti\$Ad`" --tuning `"$Cikti\$Ayar`" --models $Model --seeds $Tohum --worker $i >> `"$Cikti\${Ad}_$i.txt`" 2>&1")
        $satirlar | Set-Content $betik -Encoding ascii
        $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
            CommandLine = "cmd.exe /c `"$betik`""; CurrentDirectory = $kod; ProcessStartupInformation = $gizli }
        if ($r.ReturnValue -ne 0) { Yaz "HATA: isci $i baslatilamadi ($($r.ReturnValue))"; continue }
        $pids += $r.ProcessId
        Start-Sleep -Seconds 3
    }
    $pids | Set-Content $pidDosya
    Bekle $pidDosya
    $n = Satir "$Cikti\$Ad\metrics_w*.csv"
    Yaz "$Ad durdu: $n / $beklenen"
    if ($n -ge $beklenen) { break }
}
Yaz "$Ad zinciri bitti"
