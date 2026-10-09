# WP10 -- seeds 3,4,5 (config seeds 2,3,4), main result only.
# Per instruction: oracle-at-alpha, exhaustive block surface, NAF at K=25/100
# with the precision floor. NOT the full study.
# Every stage passes --config explicitly (see BUG_precision_floor_wrong_checkpoint.md).
$ErrorActionPreference="Continue"
$root="D:\naca"; $py="$root\venv\Scripts\python.exe"; $log="$root\runs\wp10.log"
$env:NACA_ROOT="D:\naca"; $env:TORCH_HOME="D:\naca\cache\torch"; $env:HF_HOME="D:\naca\cache\hf"
$env:MPLCONFIGDIR="D:\naca\cache\mpl"; $env:TMPDIR="D:\naca\cache\tmp"
$env:TEMP="D:\naca\cache\tmp"; $env:TMP="D:\naca\cache\tmp"; $env:PYTHONUTF8="1"
function Note($m){ Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" }
function Busy { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*naca*scripts*" }).Count -gt 0 }
function Run($name,$argl){ $w=0; while((Busy) -and $w -lt 21600){Start-Sleep -Seconds 15;$w+=15}
  Note "START $name"
  $p=Start-Process -FilePath $py -ArgumentList $argl -WorkingDirectory $root `
     -RedirectStandardOutput "$root\runs\wp10_$name.log" -RedirectStandardError "$root\runs\wp10_$name.err" `
     -WindowStyle Hidden -PassThru -Wait
  Note "END   $name  exit=$($p.ExitCode)"; return $p.ExitCode }

$lock="$root\runs\wp10.lock"
if(Test-Path $lock){$o=(Get-Content $lock -Raw).Trim(); if($o -and (Get-Process -Id $o -ErrorAction SilentlyContinue)){Note "already running";exit 0}}
$PID | Out-File -Encoding ascii $lock
try{
  Note "=== WP10 seeds 3-5 started ==="
  foreach($s in @(2,3,4)){
    $cfg="$root\configs\setup_a_seed$s.yaml"
    $tag="s$s"
    Note "--- seed config $s ---"

    Run "${tag}_train" @("-u","$root\scripts\01_train_models.py","--config",$cfg) | Out-Null

    # m=25,000 at alpha_train; 10,000 at each of the other three rates.
    Run "${tag}_collect" @("-u","$root\scripts\02_collect.py","--config",$cfg,
      "--extra-train-alphas","0.01,0.05,0.20","--extra-train-m","10000") | Out-Null

    $rd=(Get-ChildItem "$root\runs" -Directory -Filter "*_A-s${s}_collect" | Sort-Object Name | Select-Object -Last 1).FullName
    Note "seed $s run dir: $rd"
    Copy-Item "$root\runs\*_A-s${s}_train\component_index.json" $rd -ErrorAction SilentlyContinue

    Run "${tag}_fit_linear" @("-u","$root\scripts\03_fit_linear.py","--config",$cfg,"--collect-dir",$rd,"--lam","100") | Out-Null
    Run "${tag}_oracle"     @("-u","$root\scripts\wp1d_structure_and_fig13.py","--config",$cfg,"--collect-dir",$rd,"--lam","100") | Out-Null
    Run "${tag}_wp2b"       @("-u","$root\scripts\wp2b_exhaustive_coarse.py","--config",$cfg,"--collect-dir",$rd) | Out-Null
    Run "${tag}_Kcm25"      @("-u","$root\scripts\04_fit_naca.py","--config",$cfg,"--collect-dir",$rd,"--K","25","--r-U","8","--r-P","8","--tag","Kcm25","--skip-h3") | Out-Null
    Run "${tag}_Kcm100"     @("-u","$root\scripts\04_fit_naca.py","--config",$cfg,"--collect-dir",$rd,"--K","100","--r-U","30","--r-P","30","--tag","Kcm100","--skip-h3") | Out-Null
    Run "${tag}_precision"  @("-u","$root\scripts\wp0_precision_floor.py","--config",$cfg,"--collect-dir",$rd,"--n-masks","3000") | Out-Null

    Note "--- seed config $s DONE ---"
  }
  Note "=== WP10 COMPLETE ==="
} finally { Remove-Item $lock -ErrorAction SilentlyContinue }
