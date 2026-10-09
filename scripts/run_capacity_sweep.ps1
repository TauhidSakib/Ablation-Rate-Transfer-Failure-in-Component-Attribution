$ErrorActionPreference = "Continue"
$root="D:\naca"; $runDir="$root\runs\20260824_015507_A_collect"; $py="$root\venv\Scripts\python.exe"
$log="$root\runs\remaining.log"
$env:NACA_ROOT="D:\naca"; $env:TORCH_HOME="D:\naca\cache\torch"; $env:HF_HOME="D:\naca\cache\hf"
$env:MPLCONFIGDIR="D:\naca\cache\mpl"; $env:TMPDIR="D:\naca\cache\tmp"
$env:TEMP="D:\naca\cache\tmp"; $env:TMP="D:\naca\cache\tmp"; $env:PYTHONUTF8="1"
function Note($m){ Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" }
$lock="$root\runs\capsweep.lock"
if (Test-Path $lock) {
  $old=(Get-Content $lock -Raw).Trim()
  if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) { Note "capsweep already running (pid $old)"; exit 0 }
}
$PID | Out-File -Encoding ascii $lock
try {
  while (@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*naca*scripts*" }).Count -gt 0) { Start-Sleep -Seconds 15 }
  Note "START capacity_matched_sweep"
  $p = Start-Process -FilePath $py -ArgumentList "-u","$root\scripts\08_ablations.py","--collect-dir",$runDir,"--only","K_capacity_matched" `
       -WorkingDirectory $root -RedirectStandardOutput "$root\runs\capsweep.log" -RedirectStandardError "$root\runs\capsweep.err" `
       -WindowStyle Hidden -PassThru -Wait
  Note "END   capacity_matched_sweep  exit=$($p.ExitCode)"
} finally { Remove-Item $lock -ErrorAction SilentlyContinue }
