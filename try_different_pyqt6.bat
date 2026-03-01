@echo off
REM Try different PyQt6 versions to find one that works

echo ===================================================
echo PyQt6 Version Compatibility Test
echo ===================================================
echo.
echo This will try different PyQt6 versions to find
echo one that works with your Python installation.
echo.

REM Check Python version first
echo Checking Python version...
python --version
echo.

python -c "import sys; v=sys.version_info; print(f'Python {v.major}.{v.minor}.{v.micro}'); exit(0 if v.major==3 and v.minor in [9,10,11] else 1)"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Your Python version is not compatible!
    echo You MUST use Python 3.9, 3.10, or 3.11
    echo.
    echo Current version is NOT supported.
    echo Please see PYTHON_DOWNGRADE_GUIDE.txt
    echo.
    pause
    exit /b 1
)

echo Python version is compatible. Proceeding...
echo.
pause

REM Completely clean slate
echo Step 1: Removing ALL PyQt6 packages...
python -m pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip PyQt6-WebEngine PyQt6-WebEngine-Qt6 PyQt6-Charts PyQt6-Charts-Qt6
python -m pip cache purge

echo.
echo ===================================================
echo Trying PyQt6 6.4.0 (Older, more stable)
echo ===================================================
echo.

python -m pip install PyQt6==6.4.0 --no-cache-dir

echo.
echo Testing import...
python -c "import PyQt6.QtCore; print('SUCCESS! PyQt6 6.4.0 works!')"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===================================================
    echo PyQt6 6.4.0 works! Installing other dependencies...
    echo ===================================================
    python -m pip install numpy==1.26.4
    python -m pip install pyinstaller==6.0.0
    goto BUILD
)

echo.
echo PyQt6 6.4.0 failed. Trying 6.5.0...
echo.

python -m pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
python -m pip install PyQt6==6.5.0 --no-cache-dir

python -c "import PyQt6.QtCore; print('SUCCESS! PyQt6 6.5.0 works!')"
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===================================================
    echo PyQt6 6.5.0 works! Installing other dependencies...
    echo ===================================================
    python -m pip install numpy==1.26.4
    python -m pip install pyinstaller==6.0.0
    goto BUILD
)

echo.
echo PyQt6 6.5.0 failed. Trying 6.6.0...
echo.

python -m pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
python -m pip install PyQt6==6.6.0 --no-cache-dir

python -c "import PyQt6.QtCore; print('SUCCESS! PyQt6 6.6.0 works!')"
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===================================================
    echo PyQt6 6.6.0 works! Installing other dependencies...
    echo ===================================================
    python -m pip install numpy==1.26.4
    python -m pip install pyinstaller==6.0.0
    goto BUILD
)

echo.
echo PyQt6 6.6.0 failed. Trying latest version...
echo.

python -m pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
python -m pip install PyQt6 --no-cache-dir

python -c "import PyQt6.QtCore; print('SUCCESS! Latest PyQt6 works!')"
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===================================================
    echo Latest PyQt6 works! Installing other dependencies...
    echo ===================================================
    python -m pip install numpy==1.26.4
    python -m pip install pyinstaller==6.0.0
    goto BUILD
)

echo.
echo ===================================================
echo ERROR: No PyQt6 version works!
echo ===================================================
echo.
echo This suggests a deeper issue:
echo.
echo 1. Your Python installation may be corrupted
echo    Solution: Reinstall Python 3.11
echo.
echo 2. Windows system files may be missing
echo    Solution: Run "sfc /scannow" as Administrator
echo.
echo 3. Antivirus may be blocking DLL loading
echo    Solution: Temporarily disable antivirus
echo.
echo 4. Your Windows version may be too old
echo    Solution: Update to Windows 10 or 11
echo.
pause
exit /b 1

:BUILD
echo.
echo ===================================================
echo PyQt6 is working! Building application...
echo ===================================================
echo.

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build
python -m PyInstaller --clean vr180_processor.spec

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Build failed! Check errors above.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo Build completed successfully!
echo ===================================================
echo.
echo Testing the built application...
echo.

cd dist\VR180Processor
VR180Processor.exe
cd ..\..

echo.
pause
