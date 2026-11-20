# Duyarlilik: S2d2 = S2 ile ayni, ama hedef saat icin 48 saat once uretilmis ICON tahmini (S2'de 24 saat once).
# Yalniz referans yontemler (sureklilikler, klimatoloji, Ridge, LightGBM); sinir agi yok, GPU kullanilmaz.
# Isciler verim cekirdeklerine baglanir (GPU kuyrugundaki 8 performans cekirdegi iscisine dokunmaz).
#   egitim_s2d2.ps1            -> <Cikti>\ana_s2d2\metrics_w*.csv  (25 santral x 2 ufuk x 6 yontem = 300 satir)
# Baslatma WMI iledir (uygulama kapansa da surer).  Karsilastirma: kod\s2d2_tablosu.py
param([string]$Cikti = "$env:USERPROFILE\pvbench_cikti", [string]$Ad = "ana_s2d2", [int]$Isci = 8)
$py = "$env:USERPROFILE\.venvs\pvbench\Scripts\python.exe"
$kod = $PSScriptRoot
$gizli = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
New-Item -ItemType Directory -Force "$Cikti\$Ad" | Out-Null
$pids = @()
for ($i = 0; $i -lt $Isci; $i++) {
    $betik = "$Cikti\${Ad}_$i.cmd"
    @("@echo off", "set PYTHONIOENCODING=utf-8",
      "set PVBENCH_AFFINITY=$('{0:X}' -f ([int64]3 -shl (16 + 2 * $i)))",
      "cd /d `"$kod`"",
      "`"$py`" -W ignore -m pvbench.experiment --stage main --out `"$Cikti\$Ad`" --info S2d2 --horizons 1 24 --worker $i --models >> `"$Cikti\${Ad}_$i.txt`" 2>&1"
    ) | Set-Content $betik -Encoding ascii
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = "cmd.exe /c `"$betik`""; CurrentDirectory = $kod; ProcessStartupInformation = $gizli }
    if ($r.ReturnValue -eq 0) { $pids += $r.ProcessId } else { Write-Output "HATA: isci $i baslatilamadi ($($r.ReturnValue))" }
    Start-Sleep -Seconds 2
}
$pids | Set-Content "$Cikti\${Ad}_pid.txt"
Write-Output "baslatildi: $($pids.Count) isci, pid $($pids -join ' ')"
