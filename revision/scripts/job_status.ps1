param(
    [Parameter(Mandatory = $true)][string]$JobName,
    [switch]$Json
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Test-ProcessRunning([int]$PidValue) {
    if ($PidValue -le 0) { return $false }
    return [bool](Get-Process -Id $PidValue -ErrorAction SilentlyContinue)
}

function Get-RowCount([string]$CsvPath) {
    if (!(Test-Path $CsvPath)) { return 0 }
    try { return (Import-Csv $CsvPath).Count } catch { return -1 }
}

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$stateFile = Join-Path $root "revision/logs/jobs/$JobName/job_state.json"
if (!(Test-Path $stateFile)) { throw "State file not found: $stateFile" }

$state = Get-Content -Path $stateFile -Raw | ConvertFrom-Json
$rows = Get-RowCount $state.csv_path
$target = [int]$state.target_rows
$pct = if ($target -gt 0 -and $rows -ge 0) { [math]::Round($rows * 100.0 / $target, 2) } else { 0 }

$workerRunning = Test-ProcessRunning([int]$state.worker_pid)
$notifierRunning = Test-ProcessRunning([int]$state.notifier_pid)

$lastWorker = ""
if (Test-Path $state.worker_log) {
    $lastWorker = (Get-Content $state.worker_log -Tail 1 -ErrorAction SilentlyContinue)
}
$lastMilestone = ""
if (Test-Path $state.notifier_log) {
    $lastMilestone = (Get-Content $state.notifier_log -Tail 1 -ErrorAction SilentlyContinue)
}

$obj = [ordered]@{
    job_name = $state.job_name
    status = $state.status
    started_at = $state.started_at
    rows = $rows
    target_rows = $target
    percent = $pct
    worker_pid = [int]$state.worker_pid
    worker_running = $workerRunning
    notifier_pid = [int]$state.notifier_pid
    notifier_running = $notifierRunning
    csv_path = $state.csv_path
    worker_log = $state.worker_log
    worker_err_log = $state.worker_err_log
    notifier_log = $state.notifier_log
    last_worker_line = $lastWorker
    last_milestone_line = $lastMilestone
}

if ($Json) {
    $obj | ConvertTo-Json -Depth 8
} else {
    $obj.GetEnumerator() | ForEach-Object { "{0}={1}" -f $_.Key, $_.Value }
}
