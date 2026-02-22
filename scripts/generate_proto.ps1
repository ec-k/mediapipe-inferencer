# Generate Python and gRPC code from proto files
# Usage: .\scripts\generate_proto.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$ProtoDir = Join-Path $ProjectRoot "src\proto_generated"
$ProtoFile = Join-Path $ProtoDir "inferencer_control.proto"

Write-Host "Generating Python and gRPC code from: $ProtoFile"

# Generate protobuf and grpc python code using grpcio-tools via uv
uv run --group dev python -m grpc_tools.protoc `
    --proto_path=$ProtoDir `
    --python_out=$ProtoDir `
    --grpc_python_out=$ProtoDir `
    $ProtoFile

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to generate proto files"
    exit 1
}

Write-Host "Generated files successfully"

# Fix import statements in grpc file to use relative imports
$GrpcFile = Join-Path $ProtoDir "inferencer_control_pb2_grpc.py"

if (Test-Path $GrpcFile) {
    Write-Host "Fixing import statements in: $GrpcFile"

    $content = Get-Content $GrpcFile -Raw
    # Replace absolute import with relative import
    # e.g., "import inferencer_control_pb2 as" -> "from . import inferencer_control_pb2 as"
    # (?m) enables multiline mode so ^ matches start of each line
    $content = $content -replace '(?m)^import (inferencer_control_pb2) as', 'from . import $1 as'

    Set-Content -Path $GrpcFile -Value $content -NoNewline

    Write-Host "Import statements fixed"
}

Write-Host "Done!"
