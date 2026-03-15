param(
    [string]$ScoresCsv,
    [string]$CaptureDir,
    [string]$BucketColumn = "personal_bucket"
)

$rows = Import-Csv $ScoresCsv
$moved = 0
$deduped = 0
$skipped = 0

foreach ($row in $rows) {
    $source = [string]$row.filepath
    $bucket = [string]$row.$BucketColumn
    $targetDir = Join-Path $CaptureDir $bucket
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
