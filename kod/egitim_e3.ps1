# E3 (parametre eslemeli kosu) zinciri: E9 zinciri bitince kendiliginden baslar.
#   11 mimarinin hepsi yaklasik 50 000 parametreye getirilir (genislik carpani; pvbench.models.match_width).
#   Diger ayarlar (dropout, ogrenme hizi, yigin boyutu) ana kiyasta secilen degerlerde kalir; yeni ayar aramasi yok.
#   25 santral, 1 ve 24 saat, S1 (ayar secimi <Cikti>\ayar) ve S2-ICON (<Cikti>\ayar_icon), 3 tohum, referans yok.
#   Sonuclar <Cikti>\ana_e3 (bilgi kumesi basina 25 x 2 x 11 x 3 = 1650 satir); elde edilen parametre sayisi "params" sutununda.
# Baslatma: egitim_e2.ps1 ile ayni (WMI).  Gunluk: <Cikti>\e3_zincir.txt
param([string]$Cikti = "$env:USERPROFILE\pvbench_cikti", [int]$Isci = 8, [int]$Parametre = 50000, [int]$Tohum = 3,
      [string]$OncekiPid = "e9_zincir_pid.txt")
$kod = $PSScriptRoot
$gunluk = "$Cikti\e3_zincir.txt"
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

Yaz "E3 zinciri basladi; onceki zincir ($OncekiPid) bekleniyor"
$onceki = Get-Content "$Cikti\$OncekiPid" -ErrorAction SilentlyContinue
while ($onceki -and (Get-Process -Id $onceki -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 60 }
Bekle
$beklenen = 25 * 2 * 11 * $Tohum
foreach ($is in @(@("S1", "ayar"), @("S2", "ayar_icon"))) {
    foreach ($deneme in 1..3) {
        Yaz "ana_e3 $($is[0]) basliyor (deneme $deneme): hedef $Parametre parametre, tohum $Tohum"
        & "$kod\egitim_baslat.ps1" -Isci $Isci -Cikti $Cikti -Ayar $is[1] -Ana ana_e3 -Bilgi $is[0] -Tohum $Tohum -AyarYok `
            -Ek "--match-params $Parametre --no-references" | Out-Null
        Bekle
        $n = Satir "ana_e3" $is[0]
        Yaz "ana_e3 $($is[0]) durdu: $n / $beklenen"
        if ($n -ge $beklenen) { break }
    }
}
Yaz "E3 zinciri bitti"
