topdir=$1
imagename=$2 # rgba.png or image.png
for i in $topdir/*; do
    echo preprocessing "$i"/$imagename ...
    python scripts/preprocess_image.py "$i"/$imagename
done