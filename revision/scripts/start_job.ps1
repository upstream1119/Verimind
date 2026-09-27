param(
    [Parameter(Mandatory = $true)][string]$JobName,
    [Parameter(Mandatory = $true)][string]$PythonScript,
    [string[]]$ScriptArgs = @(),
    [switch]$Background,
    [string]$CsvPath = "",
    [int]$TargetRows = 0,
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

$workAbs = Resolve-ProjectPath -PathValue $WorkDir -BaseDir $WorkDir
$pythonScriptAbs = Resolve-ProjectPath -PathValue $PythonScript -BaseDir $workAbs
if (!(Test-Path $pythonScriptAbs)) { throw "Python script not found: $pythonScriptAbs" }

$jobDir = Join-Path $workAbs "revision/logs/jobs/$JobName"
New-Item -ItemType Directory -Force -Path $jobDir | Out-Null
$workerLog = Join-Path $jobDir "worker.out.log"
$workerErr = Join-Path $jobDir "worker.err.log"

if ($Background) {
    if ([string]::IsNullOrWhiteSpace($CsvPath) -or $TargetRows -le 0) {
        throw "Background mode requires -CsvPath and -TargetRows."
    }
    $csvAbs = Resolve-ProjectPath -PathValue $CsvPath -BaseDir $workAbs
    $bgScript = Join-Path $workAbs "revision/scripts/start_background_job.ps1"
    if (!(Test-Path $bgScript)) { throw "Background launcher not found: $bgScript" }

    & $bgScript `
      -JobName $JobName `
      -CsvPath $csvAbs `
      -TargetRows $TargetRows `
      -PythonScript $pythonScriptAbs `
      -ScriptArgs $ScriptArgs `
      -Milestones $Milestones `
      -PollSec $PollSec `
      -StatusEverySec $StatusEverySec `
      -PythonExe $PythonExe `
      -WorkDir $workAbs
    exit 0
}

$env:PYTHONIOENCODING = "utf-8"
if (Test-Path $workerLog) { Remove-Item -LiteralPath $workerLog -Force }
if (Test-Path $workerErr) { Remove-Item -LiteralPath $workerErr -Force }

$argList = @("-u", $pythonScriptAbs) + $ScriptArgs
Write-Host "前台模式启动：$JobName"
Write-Host "工作目录：$workAbs"
Write-Host "脚本：$pythonScriptAbs"
Write-Host "参数：$($ScriptArgs -join ' ')"
Write-Host "日志：$workerLog"

& $PythonExe $argList 2>&1 | Tee-Object -FilePath $workerLog
