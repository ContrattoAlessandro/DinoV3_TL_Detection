# Detached auto-resume watchdog for a dinov3_global training run.
#
# MUST be launched detached, otherwise the training process dies with the agent
# session shell (happened twice on this host):
#   powershell -NoProfile -Command "Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','dinov3_global\scripts\watchdog.ps1','-Run','dinov3_global\runs\exp3','-Config','dinov3_global\configs\v3.yaml' -WindowStyle Hidden"
#
# Reloads --resume last.pt if the process is killed, up to $MaxAttempts, and exits
# on its own once history.jsonl holds $Epochs completed epochs. Every action is
# appended to runs\<name>_watchdog.log so a mystery kill can be diagnosed.
param(
    [string]$Run = 'dinov3_global\runs\exp3',
    [string]$Config = 'dinov3_global\configs\v3.yaml',
    [int]$Epochs = 30,
    [int]$MaxAttempts = 20
)
$name = Split-Path $Run -Leaf
$wlog = "dinov3_global\runs\${name}_watchdog.log"

function Log($m) {
    $line = "{0} WATCHDOG[{1}]: {2}" -f (Get-Date -Format 'HH:mm:ss'), $name, $m
    Write-Output $line
    Add-Content -Path $wlog -Value $line
}

Log "started (pid $PID) run=$Run config=$Config epochs=$Epochs"
for ($i = 1; $i -le $MaxAttempts; $i++) {
    $ep = 0
    if (Test-Path "$Run\history.jsonl") {
        $ep = (Get-Content "$Run\history.jsonl" | Where-Object { $_.Trim() } | Measure-Object -Line).Lines
    }
    if ($ep -ge $Epochs) { Log "training complete ($ep epochs)"; break }
    if (Test-Path "$Run\STOPPED") { Log "early-stop marker present; not resuming"; break }
    $argList = @('dinov3_global\scripts\train.py', '--config', $Config, '--out', $Run)
    if (Test-Path "$Run\last.pt") { $argList += @('--resume', "$Run\last.pt") }
    Log "attempt $i from epoch $ep (resume=$(Test-Path "$Run\last.pt"))"
    $p = Start-Process -FilePath 'python' -ArgumentList $argList `
        -WorkingDirectory (Get-Location).Path `
        -RedirectStandardOutput "dinov3_global\runs\${name}_stdout.log" `
        -RedirectStandardError "dinov3_global\runs\${name}_console.log" `
        -WindowStyle Hidden -PassThru
    Log "python pid $($p.Id) launched"
    $p.WaitForExit()
    Log "python pid $($p.Id) exited code $($p.ExitCode)"
    Start-Sleep -Seconds 15
}
Log "watchdog exiting"
