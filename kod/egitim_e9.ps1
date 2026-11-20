# E9 (cok ufuklu egri) zinciri: E2 zinciri bitince kendiliginden baslar.
#   Ufuklar 2, 3, 6, 12, 18 saat (1 ve 24 saat ana kiyasta var); en iyi 4 model + 6 referans; 3 tohum.
#   Yeni ayar aramasi yapilmaz: her ufuk, ayni bilgi kumesi ve modelin EN YAKIN ayarli ufkunun (1 ya da 24 saat)
#   secilmis ayarini kullanir (experiment.py, jobs_for).
#   S1 -> ayar secimi <Cikti>\ayar,  S2 (ICON) -> <Cikti>\ayar_icon;  sonuclar <Cikti>\ana_e9
# Baslatma: egitim_e2.ps1 ile ayni (WMI).  Gunluk: <Cikti>\e9_zincir.txt
param([string]$Cikti = "$env:USERPROFILE\pvbench_cikti", [int]$Isci = 8, [string]$Ufuk = "2 3 6 12 18", [int]$Tohum = 3)
$kod = $PSScriptRoot
$gunluk = "$Cikti\e9_zincir.txt"
function Yaz($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Add-Content $gunluk -Encoding utf8 }
function Satir($klasor, $bilgi) {
    $n = 0
    Get-ChildItem "$Cikti\$klasor\metrics_w*.csv" -ErrorAction SilentlyContinue |
        ForEach-Object { $n += @(Import-Csv $_.FullName | Where-Object { $_.info -eq $bilgi }).Count }
    $n
}
function Bekle {
    while (@(Get-Content "$Cikti\pid.txt" -ErrorAction SilentlyContinue | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }).Count -gt 0) {
        Start-Sleep -Seconds 60
    }
}
Add-Type -Namespace Win -Name Guc -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Win.Guc]::SetThreadExecutionState(0x80000001) | Out-Null

Yaz "E9 zinciri basladi; E2 zinciri bekleniyor"
$e2 = Get-Content "$Cikti\e2_zincir_pid.txt" -ErrorAction SilentlyContinue
while ($e2 -and (Get-Process -Id $e2 -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 60 }
Bekle
if (-not (Test-Path "$Cikti\e2_modeller.json")) { Yaz "e2_modeller.json yok (E2 zinciri model secmeden bitmis). Zincir cikiyor."; exit 2 }
$modeller = ((Get-Content "$Cikti\e2_modeller.json" -Raw | ConvertFrom-Json).chosen) -join ' '
$beklenen = 25 * @($Ufuk -split ' ').Count * (6 + 4 * $Tohum)
foreach ($is in @(@("S1", "ayar"), @("S2", "ayar_icon"))) {
    foreach ($deneme in 1..3) {
        Yaz "ana_e9 $($is[0]) basliyor (deneme $deneme): ufuklar [$Ufuk], modeller [$modeller], tohum $Tohum"
        & "$kod\egitim_baslat.ps1" -Isci $Isci -Cikti $Cikti -Ayar $is[1] -Ana ana_e9 -Bilgi $is[0] -Model $modeller -Ufuk $Ufuk -Tohum $Tohum -AyarYok | Out-Null
        Bekle
        $n = Satir "ana_e9" $is[0]
        Yaz "ana_e9 $($is[0]) durdu: $n / $beklenen"
        if ($n -ge $beklenen) { break }
    }
}
Yaz "E9 zinciri bitti"
