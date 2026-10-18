# Activates the project-local toolchain for the current PowerShell session only.
# Usage:  . .\scripts\env.ps1
$root = Split-Path -Parent $PSScriptRoot
$deps = Join-Path $root ".deps"

$msvc = Get-ChildItem (Join-Path $deps "msvc\VC\Tools\MSVC") | Select-Object -First 1
$kit  = Join-Path $deps "msvc\sdkroot\Windows Kits\10"
$sdkv = (Get-ChildItem (Join-Path $kit "Include") | Select-Object -First 1).Name
$cuda = Join-Path $deps "cuda\Library"

$env:CUDA_PATH   = $cuda
$env:VSLANG      = "1033"   # nvcc parses cl's banner; localized output breaks host-compiler detection
$env:INCLUDE     = @(
    "$($msvc.FullName)\include",
    "$kit\Include\$sdkv\ucrt", "$kit\Include\$sdkv\shared",
    "$kit\Include\$sdkv\um",   "$kit\Include\$sdkv\winrt"
) -join ";"
$env:LIB         = @(
    "$($msvc.FullName)\lib\x64",
    "$kit\Lib\$sdkv\ucrt\x64", "$kit\Lib\$sdkv\um\x64"
) -join ";"
$env:PATH        = @(
    "$($msvc.FullName)\bin\Hostx64\x64",
    "$kit\bin\$sdkv\x64",
    "$cuda\bin",
    (Join-Path $deps "opencv\x64\vc17\bin"),
    (Join-Path $root ".venv\Scripts"),
    $env:PATH
) -join ";"

Write-Host "Toolchain active: MSVC $($msvc.Name), Windows SDK $sdkv, CUDA at $cuda"
