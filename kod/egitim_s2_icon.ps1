# E1 bitince S2 kosusunu ICON girdisiyle kendiliginden baslatir (bkz. makale, Yontem: bilgi kumeleri).
#   1) E1 iscilerinin bitmesini bekler; E1 tam degilse (elle durdurulduysa) HICBIR SEY YAPMADAN cikar.
#   2) Saatlik tablolari yeniden kurar (nwp_* = ICON, nwpbm_* = best_match).  Isciler calisirken yapilmaz.
#   3) S2 ayar ve ana kosusunu ayri klasorlerde baslatir: <Cikti>\ayar_icon, <Cikti>\ana_icon.
# Baslatma (arka planda):
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy','Bypass','-File','kod\egitim_s2_icon.ps1'
# Gunluk: <Cikti>\s2_icon_zincir.txt
param([string]$Cikti = "$env:USERPROFILE\pvbench_cikti", [int]$Isci = 8, [int]$E1Beklenen = 6100)
$py = "$env:USERPROFILE\.venvs\pvbench\Scripts\python.exe"
$kod = $PSScriptRoot
$gunluk = "$Cikti\s2_icon_zincir.txt"
function Yaz($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Add-Content $gunluk -Encoding utf8 }
function Satir($klasor) {
    $n = 0
    Get-ChildItem "$Cikti\$klasor\metrics_w*.csv" -ErrorAction SilentlyContinue |
        ForEach-Object { $n += [Math]::Max(0, (Get-Content $_.FullName).Count - 1) }
    $n
}
# bekleme ve tablo kurulumu sirasinda bilgisayar uykuya gecmesin
Add-Type -Namespace Win -Name Guc -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Win.Guc]::SetThreadExecutionState(0x80000001) | Out-Null

Yaz "zincir basladi; E1 iscileri bekleniyor"
while (@(Get-Content "$Cikti\pid.txt" -ErrorAction SilentlyContinue | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }).Count -gt 0) {
    Start-Sleep -Seconds 60
}
$e1 = Satir "ana"
if ($e1 -lt $E1Beklenen) {
    Yaz "E1 tamamlanmamis ($e1 / $E1Beklenen); isciler durmus. Zincir hicbir sey yapmadan cikiyor."
    exit 2
}
Yaz "E1 tamam ($e1 satir). Saatlik tablolar yeniden kuruluyor (ICON)"
Set-Location $kod
$env:PYTHONIOENCODING = "utf-8"
& $py -W ignore -m pvbench.dataset *>> $gunluk
if ($LASTEXITCODE -ne 0) { Yaz "HATA: tablo kurulumu basarisiz (cikis $LASTEXITCODE). S2 baslatilmadi."; exit 1 }
& $py -W ignore -c "import sys; sys.path.insert(0, '.'); from pvbench import dataset; p = dataset.load_plants(); bad = [n for n in p.index if 'nwpbm_cloud_cover' not in dataset.load_table(n).columns]; print('santral', len(p), 'eksik', bad); sys.exit(1 if bad or len(p) != 25 else 0)" *>> $gunluk
if ($LASTEXITCODE -ne 0) { Yaz "HATA: tablo denetimi basarisiz. S2 baslatilmadi."; exit 1 }
Yaz "tablolar tamam; S2 (ICON) ayar ve ana kosusu baslatiliyor"
& "$kod\egitim_baslat.ps1" -Isci $Isci -Cikti $Cikti -Ayar ayar_icon -Ana ana_icon -Bilgi S2 *>> $gunluk
Yaz "baslatildi"
