# Backfill seed 2 so the five-seed table is complete: it was collected for WP7,
# which needed no extra-alpha pools, so it has neither an oracle curve nor an
# exhaustive block surface. Collect the three extra rates, then run both.
$ErrorActionPreference="Continue"
$root="D:\naca"; $py="$root\venv\Scripts\python.exe"
$cfg="$root\configs\setup_a_seed1_full.yaml"; $log="$root\runs\wp10.log"
$rd="$root\runs\20260826_124436_A-s1_collect"
$env:NACA_ROOT="D:\naca"; $env:TORCH_HOME="D:\naca\cache\torch"; $env:HF_HOME="D:\naca\cache\hf"
$env:MPLCONFIGDIR="D:\naca\cache\mpl"; $env:TMPDIR="D:\naca\cache\tmp"
$env:TEMP="D:\naca\cache\tmp"; $env:TMP="D:\naca\cache\tmp"; $env:PYTHONUTF8="1"
function Note($m){ Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" }
function Run($name,$argl){ Note "START $name"
  $p=Start-Process -FilePath $py -ArgumentList $argl -WorkingDirectory $root `
     -RedirectStandardOutput "$root\runs\wp10_$name.log" -RedirectStandardError "$root\runs\wp10_$name.err" `
     -WindowStyle Hidden -PassThru -Wait
  Note "END   $name  exit=$($p.ExitCode)"; return $p.ExitCode }
Note "=== seed 2 backfill started ==="
Run "s2bf_wp2b" @("-u","$root\scripts\wp2b_exhaustive_coarse.py","--config",$cfg,"--collect-dir",$rd) | Out-Null
Run "s2bf_extra" @("-u","$root\scripts\02_collect.py","--config",$cfg,"--run-dir",$rd,
  "--extra-train-alphas","0.01,0.05,0.20","--extra-train-m","10000","--only-extra") | Out-Null
Run "s2bf_oracle" @("-u","$root\scripts\wp1d_structure_and_fig13.py","--config",$cfg,"--collect-dir",$rd,"--lam","100") | Out-Null
Note "=== seed 2 backfill COMPLETE ==="
