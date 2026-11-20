# E6: gorulmemis santral deneyi (pvbench.unseen).  Havuz modeli ortak protokollu santrallerle egitilir,
# disarida birakilan santrallerde ve kisa kayitli santrallerde sinanir.
#   Sinir aglari (GPU, P cekirdekleri), onceki zincir bitince:
#       egitim_e6.ps1 -Ad e6 -Model "gru cnn_lstm_gru" -Katman 5 -Tohum 3 -OncekiPid e5_zincir_pid.txt
#   Ridge ve LightGBM (yalniz islemci; verim cekirdeklerinde, GPU kuyruguyla ayni anda):
#       egitim_e6.ps1 -Ad e6_ref -Model "ridge lightgbm" -Katman 5 -Verim -OncekiPid ""
# -Katman 0 = santraller tek tek disarida birakilir (22 katman + tum santrallerle 1 model); K = enleme gore K grup.
# Makaledeki E6 kosusu: -Katman 5 (varsayilan), modeller gru ve cnn_lstm_gru; referanslar ridge ve lightgbm.
# Sonuc <Cikti>\<Ad>\metrics_w*.csv (uyum x 25 santral satir).  Gunluk: <Cikti>\<Ad>_zincir.txt
# Baslatma: egitim_e2.ps1 ile ayni (WMI).  -Verim kosusu pid.txt'ye dokunmaz (<Ad>_pid.txt kullanir).
param([string]$Cikti = "$env:USERPROFILE\pvbench_cikti", [string]$Ad = "e6", [int]$Isci = 8, [int]$Tohum = 3,
      [int]$Katman = 5, [string]$Model = "gru cnn_lstm_gru", [int]$Toplu = 1024,
      [string]$OncekiPid = "e5_zincir_pid.txt", [switch]$Verim)
$py = "$env:USERPROFILE\.venvs\pvbench\Scripts\python.exe"
$kod = $PSScriptRoot
$gunluk = "$Cikti\${Ad}_zincir.txt"
$pidDosya = if ($Verim) { "$Cikti\${Ad}_pid.txt" } else { "$Cikti\pid.txt" }
function Yaz($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Add-Content $gunluk -Encoding utf8 }
function Satir($desen) {
    $n = 0
    Get-ChildItem $desen -ErrorAction SilentlyContinue | ForEach-Object { $n += [Math]::Max(0, (Get-Content $_.FullName).Count - 1) }
    $n
}
function Bekle($dosya) {
    while (@(Get-Content $dosya -ErrorAction SilentlyContinue | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }).Count -gt 0) {
        Start-Sleep -Seconds 60
    }
}
Add-Type -Namespace Win -Name Guc -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Win.Guc]::SetThreadExecutionState(0x80000001) | Out-Null

Yaz "$Ad zinciri basladi; modeller [$Model], katman $Katman, onceki zincir [$OncekiPid]"
if ($OncekiPid) {
    $onceki = Get-Content "$Cikti\$OncekiPid" -ErrorAction SilentlyContinue
    while ($onceki -and (Get-Process -Id $onceki -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 60 }
}
if (-not $Verim) { Bekle $pidDosya }

$modeller = @($Model -split "\s+" | Where-Object { $_ })
$nKatman = if ($Katman -gt 0) { $Katman + 1 } else { 23 }
$nTohum = if ($Verim) { 1 } else { $Tohum }
$beklenen = $nKatman * 2 * 2 * $modeller.Count * $nTohum * 25
$gizli = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
New-Item -ItemType Directory -Force "$Cikti\$Ad" | Out-Null
foreach ($deneme in 1..3) {
    Yaz "$Ad basliyor (deneme $deneme)"
    Remove-Item "$Cikti\$Ad\claims" -Recurse -Force -ErrorAction SilentlyContinue
    $pids = @()
    for ($i = 0; $i -lt $Isci; $i++) {
        $betik = "$Cikti\${Ad}_$i.cmd"
        $satirlar = @("@echo off", "set PYTHONIOENCODING=utf-8")
        if ($Verim) { $satirlar += "set PVBENCH_AFFINITY=$('{0:X}' -f ([int64]3 -shl (16 + 2 * $i)))" }
        $satirlar += @("cd /d `"$kod`"",
          "`"$py`" -W ignore -m pvbench.unseen --out `"$Cikti\$Ad`" --runs `"$Cikti`" --folds $Katman --models $Model --seeds $Tohum --batch-size $Toplu --worker $i >> `"$Cikti\${Ad}_$i.txt`" 2>&1")
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
