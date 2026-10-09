# Waits for the Stage-3 fit to finish, then collects the extra alpha_train sets
# the multi-alpha danger baseline needs (CLAUDE.md Phase 6).
#
# Sequenced rather than parallel: two torch processes cannot coexist on this
# machine (OSError WinError 1455, paging file), and two collectors on one run
# directory is exactly the mistake the PID lock now prevents.
$ErrorActionPreference = "Stop"
$root   = "D:\naca"
$runDir = "$root\runs\YOUR_SEED1_COLLECT_DIR"
$py     = "$root\venv\Scripts\python.exe"
$done   = "$runDir\naca_summary.json"

$env:NACA_ROOT="D:\naca"; $env:TORCH_HOME="D:\naca\cache\torch"; $env:HF_HOME="D:\naca\cache\hf"
$env:MPLCONFIGDIR="D:\naca\cache\mpl"; $env:TMPDIR="D:\naca\cache\tmp"
$env:TEMP="D:\naca\cache\tmp"; $env:TMP="D:\naca\cache\tmp"; $env:PYTHONUTF8="1"

while (-not (Test-Path $done)) { Start-Sleep -Seconds 20 }
# let the fit process fully exit before claiming the GPU
while (Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $py }) {
    Start-Sleep -Seconds 15
}
Add-Content "$root\runs\watchdog.log" "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  starting extra-alpha collection for the danger baseline"

$attempt = 0
while ($attempt -lt 12) {
    $alive = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $py }
    if (-not $alive) {
        if ((Test-Path "$runDir\Y_train_a0p05.npy") -and (Test-Path "$runDir\Y_train_a0p2.npy")) {
            $s1 = (Get-Content "$runDir\Y_train_a0p05.state.json" -Raw | ConvertFrom-Json).completed
            $s2 = (Get-Content "$runDir\Y_train_a0p2.state.json"  -Raw | ConvertFrom-Json).completed
            if ($s1 -ge 20000 -and $s2 -ge 20000) {
                Add-Content "$root\runs\watchdog.log" "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  extra-alpha collection COMPLETE"
                break
            }
        }
        $attempt++
        Add-Content "$root\runs\watchdog.log" "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  extra-alpha launch attempt $attempt"
        Start-Process -FilePath $py `
            -ArgumentList "-u","$root\scripts\02_collect.py","--run-dir",$runDir,"--only-extra","--extra-train-alphas","0.05,0.20","--extra-train-m","20000" `
            -WorkingDirectory $root `
            -RedirectStandardOutput "$root\runs\collect_extra.log" `
            -RedirectStandardError  "$root\runs\collect_extra.err" `
            -WindowStyle Hidden | Out-Null
        Start-Sleep -Seconds 60
    }
    Start-Sleep -Seconds 60
}
