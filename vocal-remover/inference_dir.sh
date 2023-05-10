input_dir="examples/"
output_dir="examples_output/"

for music in "$input_dir"*
do
  python inference.py --input $music --output_dir examples_output/ --postprocess --gpu 0
done