$scoresCsv = Resolve-Path 'D:\*\Capture\scores.personal.csv'
$captureDir = Split-Path $scoresCsv -Parent
$rows = Import-Csv $scoresCsv
$moved = 0
$deduped = 0
$skipped = 0

foreach ($row in $rows) {
    $source = [System.IO.Path]::GetFullPath([string]$row.filepath)
    $bucket = [string]$row.personal_bucket
    $targetDir = Join-Path $captureDir $bucket
    $target = Join-Path $targetDir ([System.IO.Path]::GetFileName($source))

    if (-not (Test-Path $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir | Out-Null
    }

    if (Test-Path $source) {
        if (Test-Path $target) {
            Remove-Item $source -Force
            $deduped++
        }
        else {
            Move-Item $source $target
            $moved++
        }
    }
    else {
        $skipped++
    }
}

Write-Output "moved=$moved deduped=$deduped skipped=$skipped"
