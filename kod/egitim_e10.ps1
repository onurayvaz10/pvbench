# E10: ikinci test yili (pvbench.later).  Ana kiyasin modelleri, yeniden egitilmeden Aralik 2025 - Ekim 2026 doneminde puanlanir.
#   Referanslar (yalniz islemci; verim cekirdeklerinde, GPU kuyruguyla ayni anda):
#       egitim_e10.ps1 -Ad e10_ref -Verim
#   GRU (GPU, performans cekirdekleri), onceki zincir bitince:
#       egitim_e10.ps1 -Ad e10 -Model "gru" -Tohum 3 -OncekiPid e8_zincir_pid.txt
# Sonuc <Cikti>\<Ad>\later_w*.csv (22 santral x 2 ufuk x 2 bilgi kumesi x yontem x tohum x 5 donem satir).
# Gunluk: <Cikti>\<Ad>_zincir.txt.  Baslatma WMI iledir; -Verim kosusu pid.txt'ye dokunmaz (<Ad>_pid.txt kullanir).
param([string]$Cikti = "$env:USERPROFILE\pvbench_cikti", [string]$Ad = "e10", [int]$Isci = 8, [int]$Tohum = 3,
      [string]$Model = "gru", [string]$OncekiPid = "e8_zincir_pid.txt", [switch]$Verim)
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

Yaz "$Ad zinciri basladi; verim [$Verim], modeller [$Model], onceki zincir [$OncekiPid]"
if (-not $Verim) {
    $onceki = Get-Content "$Cikti\$OncekiPid" -ErrorAction SilentlyContinue
    while ($onceki -and (Get-Process -Id $onceki -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 60 }
    Bekle $pidDosya
}
$modeller = @($Model -split "\s+" | Where-Object { $_ })
$secim = if ($Verim) { "" } else { "--references --models $Model --seeds $Tohum" }
$is = if ($Verim) { 5 } else { $modeller.Count * $Tohum }                 # (santral, ufuk, bilgi) hucresi basina kosu
$beklenen = 22 * 2 * 2 * $is * 5
$gizli = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
New-Item -ItemType Directory -Force "$Cikti\$Ad" | Out-Null
foreach ($deneme in 1..3) {
    Yaz "$Ad basliyor (deneme $deneme): beklenen $beklenen satir"
    Remove-Item "$Cikti\$Ad\claims" -Recurse -Force -ErrorAction SilentlyContinue
    $pids = @()
    for ($i = 0; $i -lt $Isci; $i++) {
        $betik = "$Cikti\${Ad}_$i.cmd"
        $satirlar = @("@echo off", "set PYTHONIOENCODING=utf-8")
        if ($Verim) { $satirlar += "set PVBENCH_AFFINITY=$('{0:X}' -f ([int64]3 -shl (16 + 2 * $i)))" }
        $satirlar += @("cd /d `"$kod`"",
          "`"$py`" -W ignore -m pvbench.later --out `"$Cikti\$Ad`" --runs `"$Cikti`" --worker $i $secim >> `"$Cikti\${Ad}_$i.txt`" 2>&1")
        $satirlar | Set-Content $betik -Encoding ascii
        $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
            CommandLine = "cmd.exe /c `"$betik`""; CurrentDirectory = $kod; ProcessStartupInformation = $gizli }
        if ($r.ReturnValue -ne 0) { Yaz "HATA: isci $i baslatilamadi ($($r.ReturnValue))"; continue }
        $pids += $r.ProcessId
        Start-Sleep -Seconds 3
    }
    $pids | Set-Content $pidDosya
    Bekle $pidDosya
    $n = Satir "$Cikti\$Ad\later_w*.csv"
    Yaz "$Ad durdu: $n / $beklenen"
    if ($n -ge $beklenen) { break }
}
Yaz "$Ad zinciri bitti"
