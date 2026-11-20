# Hatali islemci cekirdeginde (mantiksal islemci 4-5; isci 2) 20 Kasim 2026 03:30'dan sonra uretilen satirlari saglam
# cekirdeklerde yeniden uretir.  Once:  python dogrulama_isci2.py hazirla <Cikti>
# Kosular asil komut satirlariyla, <Cikti>\dogrulama\<kosu> klasorlerinde ve 2 disindaki isci numaralariyla calisir
# (isci numarasi baglanilan cekirdegi belirler); biten isler atlanir, yarida kalirsa ayni komut yeniden calistirilir.
# Sonra: python dogrulama_isci2.py karsilastir <Cikti> ..\sonuclar
param([string]$Cikti = "$env:USERPROFILE\pvbench_cikti", [int[]]$Isciler = @(0, 1, 3, 4))
if ($Isciler -contains 2 -or $Isciler -contains 10) { throw "isci 2 ve 10 hatali cekirdege baglanir" }
$py = "$env:USERPROFILE\.venvs\pvbench\Scripts\python.exe"
$kod = $PSScriptRoot
$d = "$Cikti\dogrulama"
$gunluk = "$d\dogrulama_zincir.txt"
function Yaz($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Add-Content $gunluk -Encoding utf8 }
[System.Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = [IntPtr]4294967247   # 4 ve 5 disindaki islemciler

$asamalar = @(
    @{ Ad = "ana_e3 S1";  Arg = "-m pvbench.experiment --stage main --out `"$d\ana_e3`" --tuning `"$Cikti\ayar`" --trials 12 --seeds 3 --info S1 --match-params 50000 --no-references" },
    @{ Ad = "ana_e3 S2";  Arg = "-m pvbench.experiment --stage main --out `"$d\ana_e3`" --tuning `"$Cikti\ayar_icon`" --trials 12 --seeds 3 --info S2 --match-params 50000 --no-references" },
    @{ Ad = "ana_e5ma";   Arg = "-m pvbench.experiment --stage main --out `"$d\ana_e5ma`" --tuning `"$Cikti\ayar`" --trials 12 --seeds 3 --info S1a --models gru cnn_lstm_gru" },
    @{ Ad = "e5_onem";    Arg = "-m pvbench.importance --out `"$d\e5_onem`" --runs `"$Cikti`" --seeds 3" },
    @{ Ad = "e6";         Arg = "-m pvbench.unseen --out `"$d\e6`" --runs `"$Cikti`" --folds 5 --models gru cnn_lstm_gru --seeds 3 --batch-size 1024" },
    @{ Ad = "e8";         Arg = "-m pvbench.resolution --out `"$d\e8`" --tuning `"$Cikti\ayar_e2`" --models gru tcn --seeds 3" }
)
Yaz "dogrulama basladi; isciler [$Isciler]"
foreach ($a in $asamalar) {
    $klasor = ($a.Arg -replace '.*--out "([^"]+)".*', '$1')
    Remove-Item "$klasor\claims" -Recurse -Force -Confirm:$false -ErrorAction SilentlyContinue
    Yaz "$($a.Ad) basliyor"
    $surecler = foreach ($i in $Isciler) {
        Start-Process -FilePath $py -ArgumentList "-W ignore $($a.Arg) --worker $i" -WorkingDirectory $kod -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput "$d\$(($a.Ad -replace ' ', '_'))_$i.txt" -RedirectStandardError "$d\$(($a.Ad -replace ' ', '_'))_$i.err"
    }
    $surecler | Wait-Process
    Yaz "$($a.Ad) bitti; cikis kodlari [$(($surecler | ForEach-Object { $_.ExitCode }) -join ' ')]"
}
Yaz "dogrulama zinciri bitti"
