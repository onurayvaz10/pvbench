@echo off
rem PVOutput uzun cekimi: once gunluk takvim, sonra 5 dakikalik veriler.
rem Kullanim: uzun_cekim.cmd [kimlik dosyasinin yolu]
rem Kesilirse yeniden calistirmak yeterlidir; kaldigi yerden devam eder.
if not "%~1"=="" set "PVOUTPUT_KIMLIK_DOSYASI=%~1"
cd /d "%~dp0..\.."
set "LOG=veri\ham\pvoutput\cekim_gunlugu.txt"
echo ==== basladi %date% %time% >> "%LOG%"
python -u kod\veri_cekme\pvoutput_cek.py gunluk >> "%LOG%" 2>&1
python -u kod\veri_cekme\pvoutput_cek.py cek >> "%LOG%" 2>&1
echo ==== bitti %date% %time% >> "%LOG%"
