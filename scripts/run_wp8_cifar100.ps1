# WP8 -- CIFAR-100 / ResNet-18. Pre-reg: the WP8/WP9 pre-registration
# Every stage is passed --config explicitly. Omitting it on one stage can
# silently load the wrong checkpoint.
$ErrorActionPreference="Continue"
$root="D:\naca"; $py="$root\venv\Scripts\python.exe"
$cfg="$root\configs\setup_b_cifar100.yaml"; $log="$root\runs\wp8.log"
$env:NACA_ROOT="D:\naca"; $env:TORCH_HOME="D:\naca\cache\torch"; $env:HF_HOME="D:\naca\cache\hf"
$env:MPLCONFIGDIR="D:\naca\cache\mpl"; $env:TMPDIR="D:\naca\cache\tmp"
$env:TEMP="D:\naca\cache\tmp"; $env:TMP="D:\naca\cache\tmp"; $env:PYTHONUTF8="1"
function Note($m){ Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" }
function Busy { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*naca*scripts*" }).Count -gt 0 }
function Run($name,$argl){ $w=0; while((Busy) -and $w -lt 14400){Start-Sleep -Seconds 15;$w+=15}
  Note "START $name"
  $p=Start-Process -FilePath $py -ArgumentList $argl -WorkingDirectory $root `
     -RedirectStandardOutput "$root\runs\wp8_$name.log" -RedirectStandardError "$root\runs\wp8_$name.err" `
     -WindowStyle Hidden -PassThru -Wait
  Note "END   $name  exit=$($p.ExitCode)"; return $p.ExitCode }
$lock="$root\runs\wp8.lock"
if(Test-Path $lock){$o=(Get-Content $lock -Raw).Trim(); if($o -and (Get-Process -Id $o -ErrorAction SilentlyContinue)){Note "already running";exit 0}}
$PID | Out-File -Encoding ascii $lock
try{
  Note "=== WP8 CIFAR-100 started ==="

  Run "train" @("-u","$root\scripts\01_train_models.py","--config",$cfg) | Out-Null

  # One invocation: extra training rates {0.01,0.05,0.20} at m=10k, then the main
  # pool (alpha=0.10, m=25k) and the 7-point test grid at m=2k each.
  Run "collect" @("-u","$root\scripts\02_collect.py","--config",$cfg,
    "--extra-train-alphas","0.01,0.05,0.20","--extra-train-m","10000") | Out-Null

  $rd=(Get-ChildItem "$root\runs" -Directory -Filter "*_B_collect" | Sort-Object Name | Select-Object -Last 1).FullName
  Note "run dir: $rd"
  Copy-Item "$root\runs\*_B_train\component_index.json" $rd -ErrorAction SilentlyContinue

  Run "fit_linear" @("-u","$root\scripts\03_fit_linear.py","--config",$cfg,"--collect-dir",$rd,"--lam","100") | Out-Null

  # C1: oracle-at-alpha at all four rates + pooled/aggregate models scored on the
  # strictly held-out rates {0.03, 0.075, 0.15}. Also emits Var(f_M) per alpha.
  Run "wp1d_oracle" @("-u","$root\scripts\wp1d_structure_and_fig13.py","--config",$cfg,"--collect-dir",$rd,"--lam","100") | Out-Null

  # C2: mean(w) single-alpha vs pooled -- the aggregate/identifiability criterion.
  Run "wp1b_aggregate" @("-u","$root\scripts\wp1b_aggregate_nonadditivity.py","--config",$cfg,"--collect-dir",$rd,"--lam","100") | Out-Null

  # C3: exhaustive 2^9 block surface, fp32, Walsh.
  Run "wp2b_exhaustive" @("-u","$root\scripts\wp2b_exhaustive_coarse.py","--config",$cfg,"--collect-dir",$rd) | Out-Null

  # C4: capacity-matched NAF at K=25 and K=100 (rank/K ~ 0.6).
  Run "Kcm25"  @("-u","$root\scripts\04_fit_naca.py","--config",$cfg,"--collect-dir",$rd,"--K","25","--r-U","8","--r-P","8","--tag","Kcm25","--skip-h3") | Out-Null
  Run "Kcm100" @("-u","$root\scripts\04_fit_naca.py","--config",$cfg,"--collect-dir",$rd,"--K","100","--r-U","30","--r-P","30","--tag","Kcm100","--skip-h3") | Out-Null

  Run "precision" @("-u","$root\scripts\wp0_precision_floor.py","--config",$cfg,"--collect-dir",$rd,"--n-masks","3000") | Out-Null

  Note "=== WP8 COMPLETE ==="
} finally { Remove-Item $lock -ErrorAction SilentlyContinue }
