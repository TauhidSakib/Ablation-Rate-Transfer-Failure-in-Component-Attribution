# Keeps counterfactual collection alive until it finishes.
#
# The run is memory-tight on this machine (7.6 GB, pagefile deep) and has already
# died once on a 15 MB allocation. Collection checkpoints into a memmap and
# resumes from the last flushed mask, so a crash costs a restart, not data --
# this loop just performs the restart. It coexists with an already-running
# collector: it only launches when none is alive.
$ErrorActionPreference = "Stop"
$root    = "D:\naca"
$runDir  = "D:\naca\runs\YOUR_SEED1_COLLECT_DIR"
$done    = Join-Path $runDir "collect_metrics.json"
$py      = "$root\venv\Scripts\python.exe"

$env:NACA_ROOT="D:\naca"; $env:TORCH_HOME="D:\naca\cache\torch"
$env:HF_HOME="D:\naca\cache\hf"; $env:MPLCONFIGDIR="D:\naca\cache\mpl"
$env:TMPDIR="D:\naca\cache\tmp"; $env:TEMP="D:\naca\cache\tmp"
$env:TMP="D:\naca\cache\tmp"; $env:PYTHONUTF8="1"

$attempt = 0
while (-not (Test-Path $done)) {
    $alive = Get-Process python -ErrorAction SilentlyContinue |
             Where-Object { $_.Path -eq $py }
    if (-not $alive) {
        $attempt++
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Add-Content "$root\runs\watchdog.log" "$stamp  launch attempt $attempt"
        Start-Process -FilePath $py `
            -ArgumentList "-u","$root\scripts\02_collect.py","--run-dir",$runDir `
            -WorkingDirectory $root `
            -RedirectStandardOutput "$root\runs\collect_detached.log" `
            -RedirectStandardError  "$root\runs\collect_detached.err" `
            -WindowStyle Hidden | Out-Null
        Start-Sleep -Seconds 30
    }
    Start-Sleep -Seconds 60
}
Add-Content "$root\runs\watchdog.log" "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  collection COMPLETE after $attempt launch(es)"
