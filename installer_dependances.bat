@echo off
echo ========================================
echo   Installation des dependances
echo ========================================
echo.

pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERREUR: Echec de l'installation
    echo Essayez avec: py -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [OK] Toutes les dependances ont ete installees!
echo.
pause

