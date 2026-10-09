# Waits for the wp2b rerun to finish, then runs the pre-registered ablation table.
# Sequential: only one torch process fits in RAM here.
$ErrorActionPreference = "Continue"
$root   = "D:\naca"
$runDir = "$root\runs\20260824_015507_A_collect"
$py     = "$root\venv\Scripts\python.exe"
$log    = "$root\runs\chain.log"
function Note($m) { Add-Content $log "$(Get-Date -Format 'HH:mm:ss')  $m" }

$env:NACA_ROOT="D:\naca"; $env:TORCH_HOME="D:\naca\cache\torch"; $env:HF_HOME="D:\naca\cache\hf"
$env:MPLCONFIGDIR="D:\naca\cache\mpl"; $env:TMPDIR="D:\naca\cache\tmp"
$env:TEMP="D:\naca\cache\tmp"; $env:TMP="D:\naca\cache\tmp"; $env:PYTHONUTF8="1"

# wait for the severity-axis rerun to have produced its output, then for the GPU
while (-not (Test-Path "$runDir\wp2b_severity_axis.csv")) { Start-Sleep -Seconds 20 }
while (Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $py }) {
    Start-Sleep -Seconds 15
}
Note "START ablation_table"
$p = Start-Process -FilePath $py `
    -ArgumentList "-u","$root\scripts\08_ablations.py","--collect-dir",$runDir `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$root\runs\ablation_table.log" `
    -RedirectStandardError  "$root\runs\ablation_table.err" `
    -WindowStyle Hidden -PassThru -Wait
Note "END   ablation_table  exit=$($p.ExitCode)"
