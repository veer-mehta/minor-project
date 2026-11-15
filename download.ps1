# Fetches datasets, annotations and weights into data/ and weights/ (about 27 GB).
# Safe to re-run: finished items are skipped and partial downloads resume.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$drive = 'https://drive.usercontent.google.com/download?export=download&confirm=t&id='
$hico  = 'https://raw.githubusercontent.com/fredzzhang/hicodet/9dd59c1f6ee2ddc011567765d0bc644858391fea'  # version pinned by UPT: objects in COCO order
$vcoco = 'https://raw.githubusercontent.com/fredzzhang/vcoco/cb13e3d3cd74158b41acee09979e25e875c02053'

function Get-File($url, $out) {
    if (Test-Path $out) { return }
    New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
    while ($true) {
        curl.exe -f -L -C - --retry 5 -o "$out.part" $url
        if ($LASTEXITCODE -eq 0) { break }
        if ($LASTEXITCODE -eq 22) { throw "server refused $url" }  # HTTP error: retrying will not help
        Start-Sleep 5
    }
    Move-Item "$out.part" $out
}

# Downloads an archive and unpacks it; $dir is the folder the archive creates.
function Get-Archive($url, $archive, $dir) {
    if (Test-Path $dir) { return }
    Get-File $url $archive
    $parent = Split-Path $dir
    tar -xf $archive -C $parent
    if ($LASTEXITCODE -ne 0) { throw "extracting $archive failed" }
    Remove-Item $archive
}

# Small files first, so code can be tested while the images download
foreach ($f in 'instances_train2015.json', 'instances_test2015.json') { Get-File "$hico/$f" "data/hicodet/$f" }
foreach ($p in 'train', 'val', 'trainval', 'test') { Get-File "$vcoco/instances_vcoco_$p.json" "data/vcoco/instances_vcoco_$p.json" }

# Released UPT-R50 models and the fine-tuned DETR-R50 detectors they start from (UPT model zoo)
Get-File ($drive + '1lvopTC_JlGHdKOyyEZTUp2VtFfCQ-08z') 'weights/upt-r50-hicodet.pt'
Get-File ($drive + '1BQ-0tbSH7UC6QMIMMgdbNpRw2NcO8yAD') 'weights/detr-r50-hicodet.pth'
Get-File ($drive + '17Cfse3pNarVgFZd0ArMg-h6yWEYpa2NQ') 'weights/upt-r50-vcoco.pt'
Get-File ($drive + '1AIqc2LBkucBAAb_ebK9RjyNS5WmnA4HV') 'weights/detr-r50-vcoco.pth'

Get-Archive 'http://images.cocodataset.org/zips/val2014.zip'   'data/coco/val2014.zip'   'data/coco/val2014'
Get-Archive ($drive + '1dUByzVzM6z1Oq4gENa1-t0FLhr0UtDaS') 'data/hicodet/hico_20160224_det.tar.gz' 'data/hicodet/hico_20160224_det'
Get-Archive 'http://images.cocodataset.org/zips/train2014.zip' 'data/coco/train2014.zip' 'data/coco/train2014'

# V-COCO evaluation: Gupta's evaluator and the COCO annotations it reads
if (-not (Test-Path data/v-coco)) { git clone --depth 1 https://github.com/s-gupta/v-coco data/v-coco }
if (-not (Test-Path data/v-coco/data/instances_vcoco_all_2014.json)) {
    Get-File 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip' 'data/coco/annotations_trainval2014.zip'
    tar -xf data/coco/annotations_trainval2014.zip -C data/coco annotations/instances_train2014.json annotations/instances_val2014.json
    Remove-Item data/coco/annotations_trainval2014.zip
    python data/v-coco/script_pick_annotations.py data/coco/annotations
}
