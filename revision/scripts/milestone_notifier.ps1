param(
    [Parameter(Mandatory = $true)][string]$CsvPath,
    [Parameter(Mandatory = $true)][int]$TargetRows,
    [int[]]$Milestones = @(500, 1000, 1500, 2000, 2500, 3000, 9000),
    [int]$PollSec = 15,
    [int]$StatusEverySec = 300,
    [string]$LogPath = "revision/logs/milestone_notifier.log",
    [string]$StatePath = "revision/logs/milestone_notifier.state.json",
    [string]$Title = "RAG Progress",
    [bool]$ExitOnComplete = $true
)

$ErrorActionPreference = "SilentlyContinue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $LogPath -Value $line
    Write-Host $line
}

function Get-RowCount([string]$path) {
    if (!(Test-Path $path)) {
        return 0
    }
    try {
        return (Import-Csv $path).Count
    } catch {
        return -1
    }
}

function Show-Notification([string]$msg) {
    Write-Log $msg
    try {
        $wshell = New-Object -ComObject WScript.Shell
        $null = $wshell.Popup($msg, 8, $Title, 64)
    } catch {
        # no-op: keep log-only fallback
    }
}

function Save-State([hashtable]$state) {
    $json = $state | ConvertTo-Json -Depth 6
    Set-Content -Path $StatePath -Value $json -Encoding UTF8
}

$logDir = Split-Path -Parent $LogPath
$stateDir = Split-Path -Parent $StatePath
if ($logDir -and !(Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}
if ($stateDir -and !(Test-Path $stateDir)) {
    New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
}

$allMilestones = @($Milestones + $TargetRows | Where-Object { $_ -gt 0 } | Sort-Object -Unique)

$state = @{
    notified = @()
    completed = $false
}

if (Test-Path $StatePath) {
    try {
        $loaded = Get-Content -Path $StatePath -Raw | ConvertFrom-Json
        $state.notified = @($loaded.notified)
        $state.completed = [bool]$loaded.completed
    } catch {
        # keep default empty state
    }
}

Write-Log ("NOTIFIER START csv={0} target={1} poll={2}s" -f $CsvPath, $TargetRows, $PollSec)
$lastStatus = Get-Date

while ($true) {
    $rows = Get-RowCount $CsvPath
    if ($rows -ge 0) {
        $pct = if ($TargetRows -gt 0) { [math]::Round(($rows * 100.0) / $TargetRows, 2) } else { 0 }

        foreach ($m in $allMilestones) {
            if (($rows -ge $m) -and (-not ($state.notified -contains $m))) {
                Show-Notification ("Milestone reached: {0} rows (current={1}, {2}% of target {3})" -f $m, $rows, $pct, $TargetRows)
                $state.notified += $m
                Save-State $state
            }
        }

        if ((Get-Date) -ge $lastStatus.AddSeconds($StatusEverySec)) {
            Write-Log ("STATUS rows={0} target={1} pct={2}" -f $rows, $TargetRows, $pct)
            $lastStatus = Get-Date
        }

        if (($rows -ge $TargetRows) -and (-not $state.completed)) {
            Show-Notification ("Completed: rows={0} target={1}" -f $rows, $TargetRows)
            $state.completed = $true
            Save-State $state
            if ($ExitOnComplete) {
                break
            }
        }
    } else {
        Write-Log "STATUS read_error Import-Csv failed; retrying"
    }

    Start-Sleep -Seconds $PollSec
}

Write-Log "NOTIFIER EXIT"
