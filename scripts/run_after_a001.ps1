# Waits for the alpha=0.01 collection, then runs the queued analyses in order.
# Sequential because only one torch process fits in RAM on this machine.
$ErrorActionPreference = "Continue"
$root   = "D:\naca"
$runDir = "$root\runs\20260824_015507_A_collect"
$py     = "$root\venv\Scripts\python.exe"
$state  = "$runDir\Y_train_a0p01.state.json"
$log    = "$root\runs\chain.log"

$env:NACA_ROOT="D:\naca"; $env:TORCH_HOME="D:\naca\cache\torch"; $env:HF_HOME="D:\naca\cache\hf"
$env:MPLCONFIGDIR="D:\naca\cache\mpl"; $env:TMPDIR="D:\naca\cache\tmp"
$env:TEMP="D:\naca\cache\tmp"; $env:TMP="D:\naca\cache\tmp"; $env:PYTHONUTF8="1"

function Note($m) { Add-Content $log "$(Get-Date -Format 'HH:mm:ss')  $m" }

# Single-instance guard. Duplicate runners would launch duplicate torch processes,
# which on this machine means an out-of-memory crash rather than merely wasted work.
$lockFile = "$rootuns\chain.lock"
if (Test-Path $lockFile) {
    $old = Get-Content $lockFile -Raw
    if (Get-Process -Id $old -ErrorAction SilentlyContinue) {
        Note "another chain runner (pid $old) is alive; exiting"
        exit 0
    }
}
$PID | Out-File -Encoding ascii $lockFile
try {

# 1. wait for collection to finish and the process to exit
while ($true) {
    if (Test-Path $state) {
        $c = (Get-Content $state -Raw | ConvertFrom-Json).completed
        if ($c -ge 20000) { break }
    }
    Start-Sleep -Seconds 30
}
Note "alpha=0.01 collection complete"
while (Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $py }) {
    Start-Sleep -Seconds 10
}

# 2. queued analyses, in order of value
$jobs = @(
  @{ n = "wp1d_oracle_with_a001"; s = "wp1d_structure_and_fig13.py"; a = @("--collect-dir",$runDir,"--lam","100") },
  @{ n = "wp2b_exhaustive";       s = "wp2b_exhaustive_coarse.py";   a = @("--collect-dir",$runDir) },
  @{ n = "wp2a_calibration";      s = "wp2a_calibrate_granularity.py"; a = @("--collect-dir",$runDir) },
  @{ n = "precision_floor";       s = "wp0_precision_floor.py";        a = @("--collect-dir",$runDir,"--n-masks","3000") }
)
foreach ($j in $jobs) {
    Note "START $($j.n)"
    $args = @("-u","$root\scripts\$($j.s)") + $j.a
    $p = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $root `
         -RedirectStandardOutput "$root\runs\$($j.n).log" `
         -RedirectStandardError  "$root\runs\$($j.n).err" `
         -WindowStyle Hidden -PassThru -Wait
    Note "END   $($j.n)  exit=$($p.ExitCode)"
}
Note "CHAIN COMPLETE"
} finally { Remove-Item $lockFile -ErrorAction SilentlyContinue }
