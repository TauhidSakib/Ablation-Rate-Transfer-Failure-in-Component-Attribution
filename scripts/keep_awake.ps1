# Keeps the machine awake while NACA work is running, WITHOUT changing the
# user's power plan. Uses SetThreadExecutionState, which is a request that
# lapses automatically when this process exits -- nothing to undo, and the
# display is still allowed to sleep.
#
# Exits on its own once no naca python process has been alive for 5 minutes.
$sig = @"
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
"@
$es = Add-Type -MemberDefinition $sig -Name Power -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
$log = "D:\naca\runs\keepawake.log"
Add-Content $log "$(Get-Date -Format 'HH:mm:ss')  keep-awake armed (pid $PID)"
$idle = 0
while ($idle -lt 300) {
    $es::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
    $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
              Where-Object { $_.CommandLine -like "*naca*scripts*" }).Count -gt 0
    if ($busy) { $idle = 0 } else { $idle += 30 }
    Start-Sleep -Seconds 30
}
$es::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null   # release the request
Add-Content $log "$(Get-Date -Format 'HH:mm:ss')  keep-awake released (idle 5 min)"
