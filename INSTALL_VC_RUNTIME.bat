@echo off
REM Guide user to install Visual C++ Runtime

echo ===================================================
echo Visual C++ Runtime Installation Required
echo ===================================================
echo.
echo The "DLL load failed" error means you need to install
echo Microsoft Visual C++ Redistributable.
echo.
echo This is a FREE Microsoft package required by PyQt6.
echo.
echo ===================================================
echo STEP 1: Download the installer
echo ===================================================
echo.
echo Opening download page in your browser...
echo.
echo Direct link: https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.

REM Try to open the download link
start https://aka.ms/vs/17/release/vc_redist.x64.exe

echo.
echo ===================================================
echo STEP 2: Install it
echo ===================================================
echo.
echo 1. The download should start automatically
echo 2. Run the downloaded file: vc_redist.x64.exe
echo 3. Click "Install"
echo 4. Wait for installation to complete
echo 5. Click "Close"
echo.
echo ===================================================
echo STEP 3: After installation
echo ===================================================
echo.
echo 1. RESTART YOUR COMPUTER (important!)
echo 2. Come back to this folder
echo 3. Run: fix_dll_error.bat
echo.
echo ===================================================

pause

echo.
echo Did you already install it and restart?
echo.
set /p INSTALLED="Type Y if yes, N if no: "

if /i "%INSTALLED%"=="Y" (
    echo.
    echo Great! Let's verify the installation...
    echo.

    REM Check if the DLL exists
    if exist "C:\Windows\System32\vcruntime140.dll" (
        echo Visual C++ Runtime is installed!
        echo.
        echo Now let's fix the PyQt6 installation...
        pause
        call fix_dll_error.bat
    ) else (
        echo.
        echo WARNING: vcruntime140.dll not found
        echo The installation may not have completed successfully.
        echo.
        echo Please:
        echo 1. Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
        echo 2. Run the installer as Administrator
        echo 3. Restart your computer
        echo 4. Run this script again
        pause
    )
) else (
    echo.
    echo OK, please install it and restart, then run fix_dll_error.bat
    pause
)
