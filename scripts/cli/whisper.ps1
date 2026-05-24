<#
.SYNOPSIS
    Send a YouTube URL or local file to the whisper MCP server and print the
    transcript to stdout. Works from any directory — local files are streamed
    to the server as a body upload, so the file does not need to live under
    D:\whisper-data.

.PARAMETER InputPath
    Either a URL (http:// or https://) or a path to a media file.

.PARAMETER Language
    Optional ISO-639-1 language code (e.g. "en", "ru"). Auto-detect if omitted.

.PARAMETER NoCleanup
    For YouTube URLs: keep the downloaded audio on the server. Default deletes.
    (Ignored for local uploads — they are always deleted from the server.)

.PARAMETER Server
    Base URL of the whisper server. Defaults to $env:WHISPER_URL or
    http://localhost:17800.

.EXAMPLE
    whisper https://www.youtube.com/watch?v=Uu3L5KVlWoo -Language ru

.EXAMPLE
    whisper .\notes.mp4
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string] $InputPath,

    [string] $Language,

    [switch] $NoCleanup,

    [switch] $NoSave,

    [string] $Server
)

$ErrorActionPreference = 'Stop'

# Force UTF-8 so Cyrillic / CJK / emoji output doesn't render as ???.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

if (-not $Server) {
    if ($env:WHISPER_URL) { $Server = $env:WHISPER_URL } else { $Server = 'http://localhost:17800' }
}
$Server = $Server.TrimEnd('/')

$isUrl = $InputPath -match '^(?i)https?://'

function Invoke-WhisperRequest {
    param([string] $Uri, [string] $Body, [string] $ContentType, [byte[]] $Bytes)

    try {
        if ($Bytes) {
            return Invoke-RestMethod -Method Post -Uri $Uri -ContentType $ContentType -Body $Bytes
        } else {
            return Invoke-RestMethod -Method Post -Uri $Uri -ContentType $ContentType -Body $Body
        }
    }
    catch {
        $errBody = $null
        if ($_.Exception.Response) {
            try { $errBody = (New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())).ReadToEnd() } catch {}
        }
        [Console]::Error.WriteLine("whisper request failed: $($_.Exception.Message)")
        if ($errBody) { [Console]::Error.WriteLine($errBody) }
        exit 1
    }
}

if ($isUrl) {
    $endpoint = "$Server/api/transcribe/youtube"
    $cleanup = -not $NoCleanup.IsPresent
    $body = [ordered]@{ url = $InputPath; cleanup = [bool]$cleanup }
    if ($Language) { $body.language = $Language }
    $json = $body | ConvertTo-Json -Compress
    $resp = Invoke-WhisperRequest -Uri $endpoint -Body $json -ContentType 'application/json'
}
else {
    # Local file — upload bytes. Resolve relative paths against current dir.
    $resolved = Resolve-Path -LiteralPath $InputPath -ErrorAction SilentlyContinue
    if (-not $resolved) {
        [Console]::Error.WriteLine("file not found: $InputPath")
        exit 1
    }
    $file = Get-Item -LiteralPath $resolved
    $filename = [System.Uri]::EscapeDataString($file.Name)
    $endpoint = "$Server/api/transcribe/upload?filename=$filename"
    if ($Language) { $endpoint += "&language=$([System.Uri]::EscapeDataString($Language))" }

    [Console]::Error.WriteLine("[whisper] uploading $($file.FullName) ($([math]::Round($file.Length / 1MB, 2)) MB)")
    $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
    $resp = Invoke-WhisperRequest -Uri $endpoint -Bytes $bytes -ContentType 'application/octet-stream'
}

# Write-Output (not [Console]::Out) so PowerShell-native `>` and `|`
# capture the transcript properly.
Write-Output $resp.text
$lang = if ($resp.language) { $resp.language } else { '?' }
[Console]::Error.WriteLine("[whisper] lang=$lang segments=$($resp.segments) elapsed_ms=$($resp.elapsed_ms)")

# Offer to save unless -NoSave was passed or output is being piped/redirected
# via the underlying OS handle (cmd.exe `>`, bash pipe). PowerShell's own `>`
# is opaque to this check — use -NoSave when redirecting in pure PowerShell.
if (-not $NoSave -and -not [Console]::IsInputRedirected -and -not [Console]::IsOutputRedirected) {
    if ($isUrl) {
        $stem = ($InputPath -replace '^.*[/=]', '' -replace '[^A-Za-z0-9._-]', '_')
        if (-not $stem) { $stem = "transcript-$(Get-Date -Format yyyyMMdd-HHmmss)" }
    } else {
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    }
    $default = Join-Path -Path (Get-Location) -ChildPath "$stem.txt"

    [Console]::Error.Write("Save transcript to '$default'? [Y/n/path] ")
    $answer = [Console]::In.ReadLine()
    if ($null -eq $answer) { return }
    $answer = $answer.Trim()

    if ($answer -eq '' -or $answer -match '^(?i)y(es)?$') {
        $target = $default
    }
    elseif ($answer -match '^(?i)n(o)?$') {
        return
    }
    else {
        # Treat anything else as a custom path (relative resolved against $PWD).
        if ([System.IO.Path]::IsPathRooted($answer)) {
            $target = $answer
        } else {
            $target = Join-Path -Path (Get-Location) -ChildPath $answer
        }
    }

    try {
        [System.IO.File]::WriteAllText($target, $resp.text, (New-Object System.Text.UTF8Encoding($false)))
        [Console]::Error.WriteLine("[whisper] saved to $target")
    }
    catch {
        [Console]::Error.WriteLine("save failed: $($_.Exception.Message)")
        exit 1
    }
}
