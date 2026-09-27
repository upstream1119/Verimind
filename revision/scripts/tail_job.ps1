param(
    [Parameter(Mandatory = $true)][string]$JobName,
    [switch]$Err
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$jobDir = Join-Path $root "revision/logs/jobs/$JobName"
$log = if ($Err) { Join-Path $jobDir "worker.err.log" } else { Join-Path $jobDir "worker.out.log" }
if (!(Test-Path $log)) { throw "Log file not found: $log" }

Get-Content -Path $log -Wait
