# WP7 -- seed 2 at full budget. Pre-reg: results/PREREG_wp7_seed2_full.md
$ErrorActionPreference="Continue"
$root="D:\naca"; $py="$root\venv\Scripts\python.exe"
$cfg="$root\configs\setup_a_seed1_full.yaml"; $log="$root\runs\wp7.log"
$env:NACA_ROOT="D:\naca"; $env:TORCH_HOME="D:\naca\cache\torch"; $env:HF_HOME="D:\naca\cache\hf"
$env:MPLCONFIGDIR="D:\naca\cache\mpl"; $env:TMPDIR="D:\naca\cache\tmp"
$env:TEMP="D:\naca\cache\tmp"; $env:TMP="D:\naca\cache\tmp"; $env:PYTHONUTF8="1"
function Note($m){ Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" }
function Busy { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*naca*scripts*" }).Count -gt 0 }
function Run($name,$argl){ $w=0; while((Busy) -and $w -lt 14400){Start-Sleep -Seconds 15;$w+=15}
  Note "START $name"
  $p=Start-Process -FilePath $py -ArgumentList $argl -WorkingDirectory $root `
     -RedirectStandardOutput "$root\runs\wp7_$name.log" -RedirectStandardError "$root\runs\wp7_$name.err" `
     -WindowStyle Hidden -PassThru -Wait
  Note "END   $name  exit=$($p.ExitCode)"; return $p.ExitCode }
$lock="$root\runs\wp7.lock"
if(Test-Path $lock){$o=(Get-Content $lock -Raw).Trim(); if($o -and (Get-Process -Id $o -ErrorAction SilentlyContinue)){Note "already running";exit 0}}
$PID | Out-File -Encoding ascii $lock
try{
  Note "=== WP7 seed-2 full budget started ==="
  Run "collect" @("-u","$root\scripts\02_collect.py","--config",$cfg) | Out-Null
  $rd=(Get-ChildItem "$root\runs" -Directory -Filter "*_A-s1_collect" | Sort-Object Name | Select-Object -Last 1).FullName
  Note "run dir: $rd"
  Copy-Item "$root\runs\*_A-s1_train\component_index.json" $rd -ErrorAction SilentlyContinue
  Run "fit_linear" @("-u","$root\scripts\03_fit_linear.py","--config",$cfg,"--collect-dir",$rd,"--lam","100") | Out-Null
  Run "naca_base"  @("-u","$root\scripts\04_fit_naca.py","--config",$cfg,"--collect-dir",$rd,"--skip-h3") | Out-Null
  foreach($k in @(@(25,8),@(50,15),@(100,30),@(200,60))){
    Run ("Kcm"+$k[0]) @("-u","$root\scripts\04_fit_naca.py","--config",$cfg,"--collect-dir",$rd,
      "--K",[string]$k[0],"--r-U",[string]$k[1],"--r-P",[string]$k[1],"--tag",("Kcm"+$k[0]),"--skip-h3") | Out-Null
  }
  Run "precision" @("-u","$root\scripts\wp0_precision_floor.py","--config",$cfg,"--collect-dir",$rd,"--n-masks","3000") | Out-Null
  Note "=== WP7 COMPLETE ==="
} finally { Remove-Item $lock -ErrorAction SilentlyContinue }
