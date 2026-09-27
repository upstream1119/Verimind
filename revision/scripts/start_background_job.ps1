param(
    [Parameter(Mandatory = $true)][string]$JobName,
    [Parameter(Mandatory = $true)][string]$CsvPath,
    [Parameter(Mandatory = $true)][int]$TargetRows,
    [Parameter(Mandatory = $true)][string]$PythonScript,
    [string[]]$ScriptArgs = @(),
    [int[]]$Milestones = @(500, 1000, 1500, 2000, 2500, 3000, 4500, 6000, 7500, 9000),
    [int]$PollSec = 15,
    [int]$StatusEverySec = 300,
    [string]$PythonExe = "D:\anaconda\envs\rag_domestic_env\python.exe",
    [string]$WorkDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Resolve-ProjectPath([string]$PathValue, [string]$BaseDir) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $PathValue))
}

function Test-ProcessRunning([int]$PidValue) {
    if ($PidValue -le 0) { return $false }
    return [bool](Get-Process -Id $PidValue -ErrorAction SilentlyContinue)
}

$workAbs = Resolve-ProjectPath -PathValue $WorkDir -BaseDir $WorkDir
$csvAbs = Resolve-ProjectPath -PathValue $CsvPath -BaseDir $workAbs
$pythonScriptAbs = Resolve-ProjectPath -PathValue $PythonScript -BaseDir $workAbs
$notifierScript = Resolve-ProjectPath -PathValue "revision/scripts/milestone_notifier.ps1" -BaseDir $workAbs

if (!(Test-Path $pythonScriptAbs)) { throw "Python script not found: $pythonScriptAbs" }
if (!(Test-Path $notifierScript)) { throw "Notifier script not found: $notifierScript" }

$jobDir = Join-Path $workAbs "revision/logs/jobs/$JobName"
New-Item -ItemType Directory -Force -Path $jobDir | Out-Null

$workerLog = Join-Path $jobDir "worker.out.log"
$workerErr = Join-Path $jobDir "worker.err.log"
$notifierLog = Join-Path $jobDir "milestones.log"
$notifierState = Join-Path $jobDir "milestones.state.json"
$stateFile = Join-Path $jobDir "job_state.json"

if (Test-Path $stateFile) {
    $old = Get-Content -Path $stateFile -Raw | ConvertFrom-Json
    if ((Test-ProcessRunning([int]$old.worker_pid)) -or (Test-ProcessRunning([int]$old.notifier_pid))) {
        throw "Job $JobName is already running. Use job_status.ps1 or stop_background_job.ps1 first."
    }
}

foreach ($f in @($workerLog, $workerErr, $notifierLog, $notifierState)) {
    if (Test-Path $f) { Remove-Item -LiteralPath $f -Force }
}

$env:PYTHONIOENCODING = "utf-8"
$workerArgs = @("-u", $pythonScriptAbs) + $ScriptArgs
$workerProc = Start-Process -WindowStyle Hidden -FilePath $PythonExe -ArgumentList $workerArgs `
    -WorkingDirectory $workAbs -RedirectStandardOutput $workerLog -RedirectStandardError $workerErr -PassThru

$notifierArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $notifierScript,
    "-CsvPath", $csvAbs,
    "-TargetRows", "$TargetRows",
    "-PollSec", "$PollSec",
    "-StatusEverySec", "$StatusEverySec",
    "-LogPath", $notifierLog,
    "-StatePath", $notifierState,
    "-Title", "$JobName Progress"
)
if ($Milestones.Count -gt 0) {
    $notifierArgs += "-Milestones"
    $notifierArgs += ($Milestones | ForEach-Object { "$_" })
}
$notifierProc = Start-Process -WindowStyle Hidden -FilePath "C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -ArgumentList $notifierArgs -WorkingDirectory $workAbs -PassThru

$state = [ordered]@{
    job_name = $JobName
    started_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    status = "running"
    workdir = $workAbs
    csv_path = $csvAbs
    target_rows = $TargetRows
    python_exe = $PythonExe
    python_script = $pythonScriptAbs
    script_args = $ScriptArgs
    milestones = $Milestones
    worker_pid = $workerProc.Id
    notifier_pid = $notifierProc.Id
    worker_log = $workerLog
    worker_err_log = $workerErr
    notifier_log = $notifierLog
    notifier_state = $notifierState
}
$state | ConvertTo-Json -Depth 8 | Set-Content -Path $stateFile -Encoding UTF8

Write-Output "job_name=$JobName"
Write-Output "worker_pid=$($workerProc.Id)"
Write-Output "notifier_pid=$($notifierProc.Id)"
Write-Output "state_file=$stateFile"
Write-Output "status_cmd=powershell -File revision/scripts/job_status.ps1 -JobName $JobName"
