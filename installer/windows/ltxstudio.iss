; LTX Studio — installer Windows (Inno Setup 6, testuale/riproducibile in CI).
; Alternativa GUI: InstallForge (vedi installer/README.md). Inno è preferito perché scriptabile e firmabile in CI.
;
; Compila con:  ISCC.exe installer\windows\ltxstudio.iss  /DAppSrc="<cartella build Nuitka>"  /DAppVer=1.2.3
; Poi FIRMA l'installer prodotto:  python build\sign.py ... (o signtool) → evita SmartScreen/AV.
;
; L'installer NON scarica modelli: installa app+runtime e crea lo shortcut. Al PRIMO AVVIO l'app mostra il
; wizard «Setup modelli» (ui/setup/first_run.py) che chiede le cartelle, collega i modelli locali e scarica
; solo i mancanti dell'edizione.

#ifndef AppVer
  #define AppVer "0.0.0"
#endif
#ifndef AppSrc
  #define AppSrc "..\..\dist\LTXStudio"
#endif
#define AppName "LTX Studio"
#define AppExe "LTXStudio.exe"
#define AppPublisher "LTX Studio"

[Setup]
AppId={{7C3A2E10-4B9D-4E2A-9F1C-LTXSTUDIO001}
AppName={#AppName}
AppVersion={#AppVer}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputBaseFilename=LTXStudio-Setup-{#AppVer}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
; Firma: passata da build/sign.py (Azure Trusted Signing). NON incorporare credenziali qui.
PrivilegesRequired=lowest
; lowest = niente UAC: install per-utente. Il wizard modelli usa junction/hardlink (no admin) → coerente.

[Languages]
Name: "it"; MessagesFile: "compiler:Languages\Italian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#AppSrc}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Run]
; Avvio dopo l'installazione → scatta il wizard «Setup modelli» al primo avvio.
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
