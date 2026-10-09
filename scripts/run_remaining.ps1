# Runs everything left in the project, unattended, surviving session end.
#
# Sequential by necessity: only one torch process fits in 7.6 GB here. Each job
# is launched detached and waited on. A failing job does NOT abort the chain --
# later jobs may still be useful and a half-finished run is worse than a
# partly-failed one that is clearly logged.
#
# Process detection matches on COMMAND LINE, not executable path. The venv's
# python.exe is a 274 KB launcher stub that execs the base interpreter, so a
# path match only ever sees the stub and misses the worker.
$ErrorActionPreference = "Continue"
$root   = "D:\naca"
$runDir = "$root\runs\YOUR_SEED1_COLLECT_DIR"
$py     = "$root\venv\Scripts\python.exe"
$log    = "$root\runs\remaining.log"

$env:NACA_ROOT="D:\naca"; $env:TORCH_HOME="D:\naca\cache\torch"; $env:HF_HOME="D:\naca\cache\hf"
$env:MPLCONFIGDIR="D:\naca\cache\mpl"; $env:TMPDIR="D:\naca\cache\tmp"
$env:TEMP="D:\naca\cache\tmp"; $env:TMP="D:\naca\cache\tmp"; $env:PYTHONUTF8="1"

function Note($m) { Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" }

function NacaPythonRunning {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
      Where-Object { $_.CommandLine -like "*naca*scripts*" }).Count -gt 0
}

# single-instance guard
$lock = "$root\runs\remaining.lock"
if (Test-Path $lock) {
    $old = (Get-Content $lock -Raw).Trim()
    if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) {
        Note "another runner (pid $old) alive; exiting"; exit 0
    }
}
$PID | Out-File -Encoding ascii $lock
Note "=== remaining-work runner started (pid $PID) ==="

$jobs = @(
  @{ n="wp2b_corrected";  s="wp2b_exhaustive_coarse.py"; a=@("--collect-dir",$runDir) },
  @{ n="ablation_table";  s="08_ablations.py";           a=@("--collect-dir",$runDir) },
  @{ n="figures";         s="07_figures.py";             a=@("--collect-dir",$runDir) }
)

try {
    foreach ($j in $jobs) {
        # never start while another torch process holds the GPU/RAM
        $waited = 0
        while ((NacaPythonRunning) -and $waited -lt 7200) { Start-Sleep -Seconds 15; $waited += 15 }
        Note "START $($j.n)"
        $arglist = @("-u","$root\scripts\$($j.s)") + $j.a
        $p = Start-Process -FilePath $py -ArgumentList $arglist -WorkingDirectory $root `
             -RedirectStandardOutput "$root\runs\$($j.n).log" `
             -RedirectStandardError  "$root\runs\$($j.n).err" `
             -WindowStyle Hidden -PassThru -Wait
        Note "END   $($j.n)  exit=$($p.ExitCode)"
    }
    Note "=== ALL COMPLETE ==="
} finally {
    Remove-Item $lock -ErrorAction SilentlyContinue
}
