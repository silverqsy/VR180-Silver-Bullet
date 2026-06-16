; Inno Setup script for VR180 Silver Bullet 2.0 (Windows x64).
;
; Build:  ISCC.exe installer\windows.iss
; Input:  dist\VR180-Silver-Bullet-{#AppVer}-windows-x64\  (the bundle the
;         release process assembles: vr180-gui.exe + FFmpeg DLLs + README)
; Output: dist\VR180-Silver-Bullet-{#AppVer}-windows-x64-setup.exe
;
; Defaults to a PER-USER install (no UAC prompt, %LOCALAPPDATA%\Programs);
; the dialog offers "install for all users" which elevates to Program Files.
; The app's settings (%APPDATA%\VR180SilverBullet2.0) are user data and are
; deliberately NOT removed on uninstall.

#define AppVer "2.0.0"
#define BundleDir "..\dist\VR180-Silver-Bullet-" + AppVer + "-windows-x64"

[Setup]
AppId={{8E0B2C51-6C7A-4F4B-9B1E-VR180SB20}}
AppName=VR180 Silver Bullet
AppVersion={#AppVer}
AppPublisher=silverqsy
AppPublisherURL=https://github.com/silverqsy/VR180-Silver-Bullet
DefaultDirName={autopf}\VR180 Silver Bullet 2.0
DefaultGroupName=VR180 Silver Bullet
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=VR180-Silver-Bullet-{#AppVer}-windows-x64-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=VR180 Silver Bullet 2.0
; Installer wizard icon + Add/Remove Programs icon (shared app artwork).
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\vr180-gui.exe
LicenseFile=

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#BundleDir}\vr180-gui.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BundleDir}\*.dll";         DestDir: "{app}"; Flags: ignoreversion
Source: "{#BundleDir}\README.txt";    DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\VR180 Silver Bullet 2.0"; Filename: "{app}\vr180-gui.exe"
Name: "{autodesktop}\VR180 Silver Bullet 2.0"; Filename: "{app}\vr180-gui.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\vr180-gui.exe"; Description: "{cm:LaunchProgram,VR180 Silver Bullet 2.0}"; Flags: nowait postinstall skipifsilent
