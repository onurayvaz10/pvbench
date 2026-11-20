# E2 (bilgi kumesi karsilastirmasi) zinciri: S2 (ICON) kosusu bitince kendiliginden baslar.
#   1) Calisan iscilerin bitmesini bekler; ana_icon tam degilse HICBIR SEY YAPMADAN cikar.
#   2) En iyi 4 modeli secer (en_iyi_modeller.py; dogrulama hatasina gore, kural betigin basinda).
#   3) Eslenmis kosu: S0m S1m S3m (S2 ile ayni ornekler) ve S12 (S1+S2 birlesik) -> <Cikti>\ayar_e2m, <Cikti>\ana_e2m
#   4) Tam donem kosusu: S0 S3                                -> <Cikti>\ayar_e2,  <Cikti>\ana_e2
# Baslatma (baslatan uygulamadan bagimsiz, WMI ile):
#   Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine =
#     'powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File "<proje>\kod\egitim_e2.ps1"' }
# Gunluk: <Cikti>\e2_zincir.txt
param([string]$Cikti = "$env:USERPROFILE\pvbench_cikti", [int]$Isci = 8, [int]$IconBeklenen = 3050)
$py = "$env:USERPROFILE\.venvs\pvbench\Scripts\python.exe"
$kod = $PSScriptRoot
$gunluk = "$Cikti\e2_zincir.txt"
function Yaz($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Add-Content $gunluk -Encoding utf8 }
function Satir($klasor) {
    $n = 0
    Get-ChildItem "$Cikti\$klasor\metrics_w*.csv" -ErrorAction SilentlyContinue |
        ForEach-Object { $n += [Math]::Max(0, (Get-Content $_.FullName).Count - 1) }
    $n
}
function Bekle {
    while (@(Get-Content "$Cikti\pid.txt" -ErrorAction SilentlyContinue | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }).Count -gt 0) {
        Start-Sleep -Seconds 60
    }
}
function Kos($ayar, $ana, $bilgi, $modeller, $ayarBeklenen, $anaBeklenen) {
    foreach ($deneme in 1..3) {                       # yarida kalirsa (isci cokmesi) kaldigi yerden surdurulur
        Yaz "$ana basliyor (deneme $deneme): bilgi [$bilgi], modeller [$modeller]"
        & "$kod\egitim_baslat.ps1" -Isci $Isci -Cikti $Cikti -Ayar $ayar -Ana $ana -Bilgi $bilgi -Model $modeller | Out-Null
        Bekle
        $a = Satir $ayar; $b = Satir $ana
        Yaz "$ana durdu: ayar $a / $ayarBeklenen, ana $b / $anaBeklenen"
        if ($a -ge $ayarBeklenen -and $b -ge $anaBeklenen) { return }
    }
    Yaz "UYARI: $ana uc denemede tamamlanamadi; zincir siradaki adima geciyor"
}
Add-Type -Namespace Win -Name Guc -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Win.Guc]::SetThreadExecutionState(0x80000001) | Out-Null

Yaz "E2 zinciri basladi; calisan isciler bekleniyor"
Bekle
$icon = Satir "ana_icon"
if ($icon -lt $IconBeklenen) { Yaz "ana_icon tamamlanmamis ($icon / $IconBeklenen). Zincir hicbir sey yapmadan cikiyor."; exit 2 }
Set-Location $kod
$env:PYTHONIOENCODING = "utf-8"
$secim = & $py -W ignore en_iyi_modeller.py $Cikti --expected-rows $IconBeklenen 2>&1
$secim | Add-Content $gunluk -Encoding utf8
if ($LASTEXITCODE -ne 0) { Yaz "HATA: model secimi basarisiz. E2 baslatilmadi."; exit 1 }
$modeller = "$($secim[-1])".Trim()
if (@($modeller -split ' ').Count -ne 4) { Yaz "HATA: 4 model bekleniyordu, gelen: [$modeller]"; exit 1 }
Yaz "secilen modeller: $modeller"
# hucre sayilari: ayar = 6 santral x 2 ufuk x bilgi x 4 model x 12 deneme;  ana = 25 x 2 x bilgi x (6 referans + 4 x 5 tohum)
Kos "ayar_e2m" "ana_e2m" "S0m S1m S3m S12" $modeller 2304 5200
Kos "ayar_e2" "ana_e2" "S0 S3" $modeller 1152 2600
Yaz "E2 zinciri bitti"
