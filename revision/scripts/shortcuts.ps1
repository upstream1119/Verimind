param()
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")

function start_job {
    param(
        [Parameter(Mandatory = $true)][string]$JobName,
        [Parameter(Mandatory = $true)][string]$PythonScript,
        [string[]]$ScriptArgs = @(),
        [switch]$Background,
        [string]$CsvPath = "",
        [int]$TargetRows = 0
    )
    & (Join-Path $root "revision/scripts/start_job.ps1") `
      -JobName $JobName `
      -PythonScript $PythonScript `
      -ScriptArgs $ScriptArgs `
      -Background:$Background `
      -CsvPath $CsvPath `
      -TargetRows $TargetRows
}

function status_job {
    param([Parameter(Mandatory = $true)][string]$JobName)
    & (Join-Path $root "revision/scripts/job_status.ps1") -JobName $JobName
}

function tail_job {
    param(
        [Parameter(Mandatory = $true)][string]$JobName,
        [switch]$Err
    )
    & (Join-Path $root "revision/scripts/tail_job.ps1") -JobName $JobName -Err:$Err
}

function stop_job {
    param([Parameter(Mandatory = $true)][string]$JobName)
    & (Join-Path $root "revision/scripts/stop_background_job.ps1") -JobName $JobName
}

Write-Host "已加载快捷命令：start_job, status_job, tail_job, stop_job"
Write-Host "默认前台可见输出；仅在显式加 -Background 时启用后台模式。"
