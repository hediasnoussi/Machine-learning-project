@echo off
echo ========================================
echo   Dashboard Crypto - Machine Learning
echo ========================================
echo.

REM Vérifier si les modèles existent
if not exist "scaler.pkl" (
    echo [ETAPE 1/3] Entrainement des modeles...
    echo.
    python train_models.py
    if errorlevel 1 (
        echo.
        echo ERREUR: Echec de l'entrainement des modeles
        echo Verifiez que vous avez installe les dependances: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo.
    echo [OK] Modeles entraines avec succes!
    echo.
) else (
    echo [OK] Modeles deja entraines, passage a l'etape suivante...
    echo.
)

echo [ETAPE 2/3] Demarrage de l'application Flask...
echo.
echo Le dashboard sera accessible a l'adresse: http://localhost:5000
echo.
echo Appuyez sur Ctrl+C pour arreter le serveur
echo.
echo ========================================
echo.

python app.py

pause

