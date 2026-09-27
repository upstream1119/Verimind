$ErrorActionPreference = "SilentlyContinue"

$outCsv = "D:\Desktop\peng\study\project\rag_project_domestic\revision\artifacts\1_2_baseline\1_2_1_main\experiment_results_sci_v3.csv"
$workerPidFile = "D:\Desktop\peng\study\project\rag_project_domestic\revision\logs\1_2_1_full_main.pid"
$workerLog = "D:\Desktop\peng\study\project\rag_project_domestic\revision\logs\1_2_1_full_main.log"
$workerErr = "D:\Desktop\peng\study\project\rag_project_domestic\revision\logs\1_2_1_full_main.err.log"
$watchLog = "D:\Desktop\peng\study\project\rag_project_domestic\revision\logs\1_2_1_watchdog.log"
$targetRows = 3000
$checkIntervalSec = 60
$reportIntervalSec = 1800

function Get-RowCount {
    param([string]$path)
    if (!(Test-Path $path)) {
        return 0
    }
    try {
        return (Import-Csv $path).Count
    } catch {
        return -1
    }
}

function Get-WorkerProcessId {
    param([string]$pidPath)
    if (!(Test-Path $pidPath)) {
        return 0
    }
    $idRaw = (Get-Content $pidPath | Select-Object -First 1).Trim()
    if (!$idRaw) {
        return 0
    }
    try {
        return [int]$idRaw
    } catch {
        return 0
    }
}

function Is-WorkerRunning {
    param([int]$processId)
    if ($processId -le 0) {
        return $false
    }
    return [bool](Get-Process -Id $processId -ErrorAction SilentlyContinue)
}

function Start-Worker {
    param(
        [string]$outputCsv,
        [string]$logPath,
        [string]$errPath,
        [string]$pidPath
    )
    if (Test-Path $logPath) {
        Remove-Item -LiteralPath $logPath -Force
    }
    if (Test-Path $errPath) {
        Remove-Item -LiteralPath $errPath -Force
    }
    $proc = Start-Process -WindowStyle Hidden -FilePath "D:\anaconda\envs\rag_domestic_env\python.exe" -ArgumentList @("-u", "run_sci_experiment_main.py", "--output", $outputCsv) -WorkingDirectory "D:\Desktop\peng\study\project\rag_project_domestic" -RedirectStandardOutput $logPath -RedirectStandardError $errPath -PassThru
    Set-Content -Path $pidPath -Value $proc.Id
    return $proc.Id
}

Add-Content -Path $watchLog -Value ("[{0}] WATCHDOG START" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
$lastReport = Get-Date

while ($true) {
    $rows = Get-RowCount $outCsv
    $workerId = Get-WorkerProcessId $workerPidFile
    $running = Is-WorkerRunning $workerId

    if ((!$running) -and ($rows -lt $targetRows)) {
        $newId = Start-Worker -outputCsv $outCsv -logPath $workerLog -errPath $workerErr -pidPath $workerPidFile
        Add-Content -Path $watchLog -Value ("[{0}] RESTART worker_pid={1} rows={2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $newId, $rows)
        $workerId = $newId
        $running = $true
    }

    $now = Get-Date
    if (($now - $lastReport).TotalSeconds -ge $reportIntervalSec) {
        $questionCount = 0
        $latestQid = -1
        try {
            $df = Import-Csv $outCsv
            $questionCount = (($df | Select-Object -ExpandProperty question_id -Unique) | Measure-Object).Count
            $latestQid = (($df | Select-Object -ExpandProperty question_id | ForEach-Object { [int]$_ } | Measure-Object -Maximum).Maximum)
        } catch {}
        $state = if ($running) { "running" } else { "stopped" }
        Add-Content -Path $watchLog -Value ("[{0}] STATUS state={1} rows={2} questions={3} latest_qid={4}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $state, $rows, $questionCount, $latestQid)
        $lastReport = $now
    }

    if ($rows -ge $targetRows) {
        Add-Content -Path $watchLog -Value ("[{0}] COMPLETED rows={1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $rows)
        break
    }

    Start-Sleep -Seconds $checkIntervalSec
}

Add-Content -Path $watchLog -Value ("[{0}] WATCHDOG EXIT" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
