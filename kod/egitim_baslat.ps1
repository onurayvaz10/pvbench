# Egitimi bagimsiz arka plan surecleri olarak baslatir (baslatan pencere ya da uygulama kapansa da surer).
#   Asama 1 (ayar): dogrulama kumesinde hiperparametre aramasi  -> <Cikti>\<Ayar>
#   Asama 2 (ana) : ana kiyas, secilen ayarlarla, 5 tohum         -> <Cikti>\<Ana>
# Yarida kalirsa ayni komutla yeniden calistirin; biten kosular atlanir.
# Durdurmak icin:  Get-Content <Cikti>\pid.txt | ForEach-Object { taskkill /PID $_ /T /F }
#
# E1, S1 ve best_match duyarlilik kumesi S2B:  egitim_baslat.ps1 -Bilgi "S1 S2B"
# E1, S2 (ICON girdisi):                       egitim_baslat.ps1 -Ayar ayar_icon -Ana ana_icon -Bilgi S2
# (Kayitli `ana` kosusunda best_match sonuclari eski adlariyla "S2" diye durur; sonuc_tablolari.py onlari S2B yapar.)
param(
    [int]$Isci = 8,
    [string]$Cikti = "$env:USERPROFILE\pvbench_cikti",
    [int]$Deneme = 12,
    [int]$Tohum = 5,
    [string]$Ayar = "ayar",
    [string]$Ana = "ana",
    [string]$Bilgi = "",         # bos: kodun varsayilani (S1 S2); ornek: "S2" ya da "S0 S3"
    [string]$Model = "",         # bos: 11 modelin hepsi; ornek: "gru cnn_lstm tcn transformer"
    [string]$Ufuk = "",          # bos: 1 ve 24 saat; ornek: "2 3 6 12 18"
    [string]$Ek = "",            # ana asamaya eklenecek secenekler; ornek: "--match-params 50000 --no-references"
    [switch]$AyarYok,            # ayar asamasi kosulmaz; <Ayar> klasorundeki hazir secim (selected.csv) kullanilir
    [int]$EkIsci = 0,            # verimli (E) cekirdeklerde calisan ek isciler; her biri 4 E-cekirdege baglanir (en cok 4)
    [switch]$YalnizEk            # calisan bir kosuya yalniz ek iscileri ekler (is paylasim kayitlari silinmez)
)
if ($EkIsci -gt 4) { throw "EkIsci en cok 4 olabilir (16 verimli cekirdek / 4)" }
$py = "$env:USERPROFILE\.venvs\pvbench\Scripts\python.exe"
$kod = $PSScriptRoot
$bilgi = if ($Bilgi) { " --info $Bilgi" } else { "" }
if ($Model) { $bilgi += " --models $Model" }
if ($Ufuk) { $bilgi += " --horizons $Ufuk" }
if ($AyarYok -and -not (Test-Path "$Cikti\$Ayar\selected.csv")) { throw "-AyarYok icin $Cikti\$Ayar\selected.csv gerekli" }
New-Item -ItemType Directory -Force "$Cikti\$Ayar", "$Cikti\$Ana" | Out-Null
if (-not $YalnizEk) {
    Remove-Item "$Cikti\$Ana\claims" -Recurse -Force -ErrorAction SilentlyContinue
    if (-not $AyarYok) { Remove-Item "$Cikti\$Ayar\claims" -Recurse -Force -ErrorAction SilentlyContinue }
}
$pids = @()
$gizli = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$ilk = if ($YalnizEk) { $Isci } else { 0 }
for ($i = $ilk; $i -lt $Isci + $EkIsci; $i++) {
    $gunluk = "$Cikti\isci_$i.txt"
    $betik = "$Cikti\isci_$i.cmd"
    $satirlar = @("@echo off", "set PYTHONIOENCODING=utf-8", "cd /d `"$kod`"")
    # ek isciler: mantiksal islemci 16-31 (verimli cekirdekler), dorderli gruplar
    if ($i -ge $Isci) { $satirlar += "set PVBENCH_AFFINITY={0:X}" -f ([int64]0xF -shl (16 + 4 * ($i - $Isci))) }
    if (-not $AyarYok) {
        $satirlar += "`"$py`" -W ignore -m pvbench.experiment --stage tune --out `"$Cikti\$Ayar`" --trials $Deneme$bilgi --worker $i >> `"$gunluk`" 2>&1"
        $satirlar += "if errorlevel 1 exit /b 1"
    }
    $satirlar += "`"$py`" -W ignore -m pvbench.experiment --stage main --out `"$Cikti\$Ana`" --tuning `"$Cikti\$Ayar`" --trials $Deneme --seeds $Tohum$bilgi$(if ($Ek) { " $Ek" }) --worker $i >> `"$gunluk`" 2>&1"
    $satirlar | Set-Content $betik -Encoding ascii
    # WMI ile baslatilir: surec, baslatan uygulamanin alt sureci OLMAZ; Start-Process ile baslatilan isciler
    # baslatan uygulama kapaninca onunla birlikte kapanir.
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = "cmd.exe /c `"$betik`""; CurrentDirectory = $kod; ProcessStartupInformation = $gizli }
    if ($r.ReturnValue -ne 0) { throw "isci $i baslatilamadi (WMI donus degeri $($r.ReturnValue))" }
    $pids += $r.ProcessId
    Start-Sleep -Seconds 3
}
if ($YalnizEk) { $pids | Add-Content "$Cikti\pid.txt" } else { $pids | Set-Content "$Cikti\pid.txt" }
"Baslatildi: $(@($pids).Count) isci, PID: $($pids -join ', ')"
