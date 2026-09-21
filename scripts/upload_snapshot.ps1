param([string]$RequestPath, [string]$ZipPath, [string]$ResultPath)
$ErrorActionPreference = 'Stop'
$req = Get-Content -LiteralPath $RequestPath -Raw | ConvertFrom-Json
$client = [System.Net.Http.HttpClient]::new()
try {
    foreach ($h in $req.upload_headers.PSObject.Properties) {
        $client.DefaultRequestHeaders.TryAddWithoutValidation($h.Name, [string]$h.Value) | Out-Null
    }
    $form = [System.Net.Http.MultipartFormDataContent]::new()
    $part = [System.Net.Http.ByteArrayContent]::new([System.IO.File]::ReadAllBytes((Resolve-Path $ZipPath)))
    $part.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::new('application/zip')
    $form.Add($part, 'file', $req.filename)
    $response = $client.PostAsync($req.upload_url, $form).GetAwaiter().GetResult()
    $response.EnsureSuccessStatusCode() | Out-Null
    $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    Set-Content -LiteralPath $ResultPath -Value $body -Encoding utf8
    ($body | ConvertFrom-Json).status
} finally { $client.Dispose() }
