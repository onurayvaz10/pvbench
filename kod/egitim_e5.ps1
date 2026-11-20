# E5 zinciri: onceki zincir (varsayilan E3) bitince kendiliginden baslar.
#   1) Hareketli ortalama ablasyonu: bilgi kumesi S1a = S1 + guc serisinin geriye donuk 3 saatlik ortalamasi
#      (calismanin ilk surumundeki ek girdi).  GRU ve hibrit CNN-LSTM-GRU (3 tohum) + 6 referans; ayarlar S1'in secilmis ayarlari.
#      Sonuc <Cikti>\ana_e5ma (25 x 2 x (6 + 2 x 3) = 600 satir); S1 ile ayni ornekler, karsilastirma ana kiyasin S1 sonuclariyla.
#   2) Permutasyon onemi (pvbench.importance): S1 ve S2, h = 1 ve 24, 25 santral; Ridge, LightGBM, GRU ve hibrit (3 tohum).
#      Sonuc <Cikti>\e5_onem\importance_w*.csv (25 x 2 x 2 x 8 model kosusu x 7 grup = 5600 satir).
# Baslatma: egitim_e2.ps1 ile ayni (WMI).  Gunluk: <Cikti>\e5_zincir.txt
param([string]$Cikti = "$env:USERPROFILE\pvbench_cikti", [int]$Isci = 8, [int]$Tohum = 3, [string]$OncekiPid = "e3_zincir_pid.txt")
$py = "$env:USERPROFILE\.venvs\pvbench\Scripts\python.exe"
$kod = $PSScriptRoot
$gunluk = "$Cikti\e5_zincir.txt"
function Yaz($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Add-Content $gunluk -Encoding utf8 }
function Satir($desen) {
    $n = 0
    Get-ChildItem $desen -ErrorAction SilentlyContinue | ForEach-Object { $n += [Math]::Max(0, (Get-Content $_.FullName).Count - 1) }
    $n
}
function Bekle {
    while (@(Get-Content "$Cikti\pid.txt" -ErrorAction SilentlyContinue | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }).Count -gt 0) {
        Start-Sleep -Seconds 60
    }
}
Add-Type -Namespace Win -Name Guc -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Win.Guc]::SetThreadExecutionState(0x80000001) | Out-Null

Yaz "E5 zinciri basladi; onceki zincir ($OncekiPid) bekleniyor"
$onceki = Get-Content "$Cikti\$OncekiPid" -ErrorAction SilentlyContinue
while ($onceki -and (Get-Process -Id $onceki -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 60 }
Bekle

$beklenen = 25 * 2 * (6 + 2 * $Tohum)
foreach ($deneme in 1..3) {
    Yaz "ana_e5ma basliyor (deneme $deneme): S1a, gru + cnn_lstm_gru, tohum $Tohum"
    & "$kod\egitim_baslat.ps1" -Isci $Isci -Cikti $Cikti -Ayar ayar -Ana ana_e5ma -Bilgi S1a -Model "gru cnn_lstm_gru" -Tohum $Tohum -AyarYok | Out-Null
    Bekle
    $n = Satir "$Cikti\ana_e5ma\metrics_w*.csv"
    Yaz "ana_e5ma durdu: $n / $beklenen"
    if ($n -ge $beklenen) { break }
}

$beklenen = 25 * 2 * 2 * (2 + 2 * $Tohum) * 7
$gizli = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
New-Item -ItemType Directory -Force "$Cikti\e5_onem" | Out-Null
foreach ($deneme in 1..3) {
    Yaz "e5_onem basliyor (deneme $deneme)"
    Remove-Item "$Cikti\e5_onem\claims" -Recurse -Force -ErrorAction SilentlyContinue
    $pids = @()
    for ($i = 0; $i -lt $Isci; $i++) {
        $betik = "$Cikti\onem_$i.cmd"
        @("@echo off", "set PYTHONIOENCODING=utf-8", "cd /d `"$kod`"",
          "`"$py`" -W ignore -m pvbench.importance --out `"$Cikti\e5_onem`" --runs `"$Cikti`" --seeds $Tohum --worker $i >> `"$Cikti\onem_$i.txt`" 2>&1") |
            Set-Content $betik -Encoding ascii
        $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
            CommandLine = "cmd.exe /c `"$betik`""; CurrentDirectory = $kod; ProcessStartupInformation = $gizli }
        if ($r.ReturnValue -ne 0) { Yaz "HATA: onem iscisi $i baslatilamadi ($($r.ReturnValue))"; continue }
        $pids += $r.ProcessId
        Start-Sleep -Seconds 3
    }
    $pids | Set-Content "$Cikti\pid.txt"
    Bekle
    $n = Satir "$Cikti\e5_onem\importance_w*.csv"
    Yaz "e5_onem durdu: $n / $beklenen"
    if ($n -ge $beklenen) { break }
}
Yaz "E5 zinciri bitti"
