# 实验运行工具（最终版）

## 目标
- 默认前台可见输出，避免“盲跑”。
- 后台模式仅在显式要求时启用。

## 一次性准备（每个新 PowerShell 会话执行一次）
```powershell
. .\revision\scripts\shortcuts.ps1
```

## 默认推荐：前台模式（实时输出）
```powershell
start_job -JobName "<任务名>" -PythonScript "<脚本路径>" -ScriptArgs @("--arg1","value1")
```

## 可选：后台模式（仅显式启用）
```powershell
start_job -JobName "<任务名>" -PythonScript "<脚本路径>" -ScriptArgs @("--arg1","value1") -Background -CsvPath "<输出CSV路径>" -TargetRows <目标行数>
```

## 后台任务管理（按需）
```powershell
status_job -JobName "<任务名>"
tail_job   -JobName "<任务名>"
stop_job   -JobName "<任务名>"
```

## 产物目录
- `revision/logs/jobs/<任务名>/worker.out.log`
- `revision/logs/jobs/<任务名>/worker.err.log`
- `revision/logs/jobs/<任务名>/milestones.log`（后台模式）
- `revision/logs/jobs/<任务名>/job_state.json`（后台模式）

## 结论
- 日常实验：用前台模式。
- 过夜长跑或你明确要求：用后台模式。
