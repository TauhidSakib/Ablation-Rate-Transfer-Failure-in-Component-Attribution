# WP5 -- seed-2 replication, end to end, unattended.
# Pre-registration: the WP5 pre-registration (written in advance).
# Nothing here is tuned to make seed 2 agree with seed 1.
$ErrorActionPreference = "Continue"
$root="D:\naca"; $py="$root\venv\Scripts\python.exe"
$cfg="$root\configs\setup_a_seed1.yaml"; $log="$root\runs\wp5.log"
$env:NACA_ROOT="D:\naca"; $env:TORCH_HOME="D:\naca\cache\torch"; $env:HF_HOME="D:\naca\cache\hf"
$env:MPLCONFIGDIR="D:\naca\cache\mpl"; $env:TMPDIR="D:\naca\cache\tmp"
$env:TEMP="D:\naca\cache\tmp"; $env:TMP="D:\naca\cache\tmp"; $env:PYTHONUTF8="1"
function Note($m){ Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" }
function Busy { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*naca*scripts*" }).Count -gt 0 }
function Run($name, $argl) {
  $w=0; while ((Busy) -and $w -lt 10800) { Start-Sleep -Seconds 15; $w+=15 }
  Note "START $name"
  $p = Start-Process -FilePath $py -ArgumentList $argl -WorkingDirectory $root `
       -RedirectStandardOutput "$root\runs\wp5_$name.log" -RedirectStandardError "$root\runs\wp5_$name.err" `
       -WindowStyle Hidden -PassThru -Wait
  Note "END   $name  exit=$($p.ExitCode)"
  return $p.ExitCode
}
$lock="$root\runs\wp5.lock"
if (Test-Path $lock) { $o=(Get-Content $lock -Raw).Trim(); if ($o -and (Get-Process -Id $o -ErrorAction SilentlyContinue)) { Note "already running (pid $o)"; exit 0 } }
$PID | Out-File -Encoding ascii $lock
try {
  Note "=== WP5 seed-2 replication started ==="
  if ((Run "train" @("-u","$root\scripts\01_train_models.py","--config",$cfg)) -ne 0) { Note "TRAIN GATE FAILED - stopping"; exit 1 }

  # find the run dir the collector will create/use
  Run "collect_main" @("-u","$root\scripts\02_collect.py","--config",$cfg) | Out-Null
  $rd = (Get-ChildItem "$root\runs" -Directory -Filter "*_A-s1_collect" | Sort-Object Name | Select-Object -Last 1).FullName
  Note "run dir: $rd"
  Run "collect_extra" @("-u","$root\scripts\02_collect.py","--config",$cfg,"--run-dir",$rd,"--only-extra","--extra-train-alphas","0.01,0.05,0.20","--extra-train-m","10000") | Out-Null

  Copy-Item "$root\runs\*_A-s1_train\component_index.json" $rd -ErrorAction SilentlyContinue

  Run "fit_linear" @("-u","$root\scripts\03_fit_linear.py","--config",$cfg,"--collect-dir",$rd,"--lam","100") | Out-Null
  Run "naca_main"  @("-u","$root\scripts\04_fit_naca.py","--config",$cfg,"--collect-dir",$rd,"--skip-h3") | Out-Null
  # claim 1: oracle-at-alpha
  Run "oracle"     @("-u","$root\scripts\wp1d_structure_and_fig13.py","--config",$cfg,"--collect-dir",$rd,"--lam","100") | Out-Null
  # claim 2: exhaustive block surface
  Run "exhaustive" @("-u","$root\scripts\wp2b_exhaustive_coarse.py","--config",$cfg,"--collect-dir",$rd,"--skip-layer") | Out-Null
  # claim 3: NAF at K=25 and K=100, matched capacity (r = 0.3K)
  Run "K25"  @("-u","$root\scripts\04_fit_naca.py","--config",$cfg,"--collect-dir",$rd,"--K","25","--r-U","8","--r-P","8","--tag","Kcm25","--skip-h3") | Out-Null
  Run "K100" @("-u","$root\scripts\04_fit_naca.py","--config",$cfg,"--collect-dir",$rd,"--K","100","--r-U","30","--r-P","30","--tag","Kcm100","--skip-h3") | Out-Null
  Note "=== WP5 COMPLETE ==="
} finally { Remove-Item $lock -ErrorAction SilentlyContinue }
