# Butun sonuc tablolarini, makalenin tablolarini ve sekillerini kayitli kosu ciktilarindan yeniden uretir
# (egitim yapmaz, GPU kullanmaz).
#   hepsini_uret.ps1                        sonuclar\, makale\tablolar\, makale\sekiller\
#   hepsini_uret.ps1 -Sonuclar <klasor> -YalnizTablolar
#                                           yalniz sonuc tablolari, baska bir klasore (kayitli tablolarla karsilastirmak icin)
# Sure olcumu (sure_olcumu.py) burada calistirilmaz: GPU ister ve sure olcer; sonucu sonuclar\sure_olcumu.csv dosyasidir.
# Sira bagimliliklara goredir: once ana kiyas ozeti, sonra onu okuyan tablolar.
param(
    [string]$Cikti = "$env:USERPROFILE\pvbench_cikti",
    [string]$Sonuclar = "",
    [switch]$YalnizTablolar
)
$ErrorActionPreference = "Stop"
$py = "$env:USERPROFILE\.venvs\pvbench\Scripts\python.exe"
$kod = $PSScriptRoot
$kok = Split-Path $kod -Parent
if (-not $Sonuclar) { $Sonuclar = "$kok\sonuclar" }
New-Item -ItemType Directory -Force $Sonuclar | Out-Null
$env:CUDA_VISIBLE_DEVICES = ""

function Calistir([string]$betik, [string[]]$arglar) {
    $basla = Get-Date
    & $py -W ignore "$kod\$betik" @arglar | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "$betik basarisiz (cikis kodu $LASTEXITCODE)" }
    "{0,-34} {1,6:N0} s" -f $betik, ((Get-Date) - $basla).TotalSeconds
}

$iki = @($Cikti, $Sonuclar)      # <cikti> <sonuclar>
$bir = @($Sonuclar)              # <sonuclar>
Calistir "sonuc_tablolari.py" $iki              # E1: ozet, santral duzeyi puanlar
Calistir "istatistik.py" $iki                   # E4: Friedman/Nemenyi, Wilcoxon (Holm), Diebold-Mariano
Calistir "duyarlilik_ortak_protokol.py" $iki    # kisa kayitli uc santral olmadan
Calistir "tablo_santraller.py" $iki             # santral envanteri (Tablo 1, Ek Tablo S1)
Calistir "hizalama_denetimi.py" $bir            # zaman hizalamasi ve konum (Ek Tablo S2)
Calistir "duzey_kaymasi.py" $bir                # egitim-test duzey kaymasi (Ek Tablo S7)
Calistir "katmanli_analiz.py" $bir              # E7
Calistir "yanlilik_konum.py" $iki               # E7: ortalama yanlilik ve konum
Calistir "e2_tablolari.py" $iki                 # E2: bilgi kumeleri
Calistir "s2d2_tablosu.py" $iki                 # tahmin yasi (S2d2)
Calistir "ek_tablolar.py" $iki                  # butun olcutler, MAPE esikleri
Calistir "mimari_tablolari.py" $iki             # katman tanimlari, parametre sayilari
Calistir "e3_tablolari.py" $iki                 # E3: esit boyut
Calistir "e5_tablolari.py" $iki                 # E5: girdi onemi, hareketli ortalama
Calistir "e6_tablolari.py" $iki                 # E6: gorulmemis santral
Calistir "e8_tablolari.py" $iki                 # E8: alt-saatlik cozunurluk
Calistir "e9_tablolari.py" $iki                 # E9: ufuklar
Calistir "ikinci_yil_tablosu.py" $iki           # E10: ikinci test yili
Calistir "ham_sureklilik.py" $bir               # son islemin referans yontemlere etkisi
Calistir "saat_siniri_denetimi.py" $bir         # PVOutput kayitlarinda saat siniri
Calistir "epias_denetimi.py" $bir               # olculen santralin ham kaydi
Calistir "dislama_denetimi.py" $bir             # dislanan sistemler icin Ek Not 1 sayilari
if ($YalnizTablolar) { "tablolar tamam: $Sonuclar"; return }

Calistir "makale_tablolari.py" @($Sonuclar, "$kok\makale\tablolar")
Calistir "sekiller.py" @($Cikti, $Sonuclar, "$kok\makale\sekiller")
"hepsi tamam"
