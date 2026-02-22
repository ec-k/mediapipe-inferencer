# Download MediaPipe models for inference scripts
# Usage: .\download_models.ps1

$ErrorActionPreference = "Stop"

$modelsDir = Join-Path $PSScriptRoot "models"

# Create models directory if not exists
if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
    Write-Host "Created directory: $modelsDir"
}

$models = @(
    @{
        Name = "pose_landmarker_full.task"
        Url  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    },
    @{
        Name = "hand_landmarker.task"
        Url  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    },
    @{
        Name = "face_landmarker.task"
        Url  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    }
)

foreach ($model in $models) {
    $outputPath = Join-Path $modelsDir $model.Name

    if (Test-Path $outputPath) {
        Write-Host "Already exists: $($model.Name)"
        continue
    }

    Write-Host "Downloading: $($model.Name)..."
    try {
        Invoke-WebRequest -Uri $model.Url -OutFile $outputPath
        Write-Host "Downloaded: $($model.Name)"
    }
    catch {
        Write-Error "Failed to download $($model.Name): $_"
        exit 1
    }
}

Write-Host "All models downloaded to: $modelsDir"
