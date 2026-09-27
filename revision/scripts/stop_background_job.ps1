param(
    [Parameter(Mandatory = $true)][string]$JobName
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Stop-IfRunning([int]$PidValue, [string]$Name) {
    if ($PidValue -le 0) { return "$Name=not_set" }
    if (Get-Process -Id $PidValue -ErrorAction SilentlyContinue) {
        Stop-Process -Id $PidValue -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
        $alive = [bool](Get-Process -Id $PidValue -ErrorAction SilentlyContinue)
        return "$Name`_stopped=$([bool](-not $alive))"
    }
    return "$Name=not_running"
}

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$stateFile = Join-Path $root "revision/logs/jobs/$JobName/job_state.json"
if (!(Test-Path $stateFile)) { throw "State file not found: $stateFile" }

$state = Get-Content -Path $stateFile -Raw | ConvertFrom-Json
$workerMsg = Stop-IfRunning -PidValue ([int]$state.worker_pid) -Name "worker"
$notifierMsg = Stop-IfRunning -PidValue ([int]$state.notifier_pid) -Name "notifier"

$state.status = "stopped_by_user"
$state.stopped_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
$state | ConvertTo-Json -Depth 8 | Set-Content -Path $stateFile -Encoding UTF8

Write-Output "job_name=$JobName"
Write-Output $workerMsg
Write-Output $notifierMsg
Write-Output "state_file=$stateFile"
