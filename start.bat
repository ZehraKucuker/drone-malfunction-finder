@echo off
echo ==========================================
echo DroneAI Sense - Baslat
echo ==========================================
echo.

REM RethinkDB Docker kontrolu
echo [1/4] RethinkDB Docker kontrol ediliyor...
docker ps | findstr "drone_rethinkdb" >nul 2>&1
if errorlevel 1 (
    echo RethinkDB baslatiliyor...
    docker-compose -f docker-compose.dev.yml up -d
    timeout /t 5 /nobreak >nul
) else (
    echo RethinkDB zaten calisiyor.
)
echo.

REM ML Servisi
echo [2/4] ML Servisi baslatiliyor...
start "ML Service" cmd /k "cd /d %~dp0drone_fault_detection_api && python ml_service.py"
timeout /t 10 /nobreak >nul
echo.

REM API Servisi
echo [3/4] API Servisi baslatiliyor...
start "API Service" cmd /k "cd /d %~dp0drone_fault_detection_api && npm start"
timeout /t 5 /nobreak >nul
echo.

REM Web Arayuzu
echo [4/4] Web arayuzu baslatiliyor...
start "Web App" cmd /k "cd /d %~dp0drone_fault_detection_web && npm run dev"
echo.

echo ==========================================
echo Tum servisler baslatildi!
echo ==========================================
echo.
echo RethinkDB Admin: http://localhost:8080
echo ML Servisi:      http://localhost:5000
echo API:             http://localhost:3001
echo Web Arayuzu:     http://localhost:3000
echo.
echo Durdurmak icin terminal pencerelerini kapatin.
pause
