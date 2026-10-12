Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | ForEach-Object {
    $cl = if ($_.CommandLine) { $_.CommandLine.Substring(0, [Math]::Min(140, $_.CommandLine.Length)) } else { '' }
    Write-Output ("{0} started={1} :: {2}" -f $_.ProcessId, $_.CreationDate.ToString('HH:mm:ss'), $cl)
}
