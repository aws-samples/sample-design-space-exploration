# Download trained model .pt files from S3 into the local git repo
# Run this AFTER the EC2 test succeeds

$S3_BUCKET = "car-design-explorer-models-$((Get-STSCallerIdentity).Account)"
$WEIGHTS_DIR = "car-design-explorer/backend/models/weights"

Write-Host "Downloading model files from S3 to local repo..."

aws s3 cp "s3://$S3_BUCKET/kpi/best_model.pt" "$WEIGHTS_DIR/kpi/best_model.pt"
aws s3 cp "s3://$S3_BUCKET/surface/best_model.pt" "$WEIGHTS_DIR/surface/best_model.pt"
aws s3 cp "s3://$S3_BUCKET/slices/ae_best_model.pt" "$WEIGHTS_DIR/slices/ae_best_model.pt"
aws s3 cp "s3://$S3_BUCKET/slices/mgn_last_model.pt" "$WEIGHTS_DIR/slices/mgn_last_model.pt"

Write-Host ""
Write-Host "Model files in repo:"
Get-ChildItem -Recurse "$WEIGHTS_DIR" -Filter "*.pt" | ForEach-Object {
    $sizeMB = [math]::Round($_.Length / 1MB, 1)
    Write-Host "  $($_.FullName) ($sizeMB MB)"
}

# Also download test output samples (VTP/PNG) for local inspection
Write-Host ""
Write-Host "Downloading test output samples..."
aws s3 sync "s3://$S3_BUCKET/test_outputs/" "car-design-explorer/infra/seed/test_outputs/" --exclude "*.pt" --exclude "*.npy"

Write-Host ""
Write-Host "Done. Check car-design-explorer/infra/seed/test_outputs/ for sample VTP/PNG files."
