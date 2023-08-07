

examples=(
	'data/nerf4/chair'
	'data/nerf4/drums'
	'data/nerf4/ficus'
	'data/nerf4/mic'
)

for i in "${examples[@]}"; do
	filename=$(basename "$i")
	bash scripts/texural_inversion/textural_inversion.sh 0 runwayml/stable-diffusion-v1-5 "$i"/rgba.png out/texural_inversion/${filename} _nerf_${filename}_ ${filename} --max_train_steps 3000
done