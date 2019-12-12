python train.py --which_loss pixel --checkpoint_dir pixelnet-1500 --max_steps 1500
python train.py --which_loss vgg --checkpoint_dir vggnet-1500 --max_steps 1500
# python train.py --which_color_pred all --checkpoint_dir allcolornet --max_steps 5
# python train.py --which_color_pred fgbg --checkpoint_dir fgbgnet --max_steps 5

python test.py --model_name pixelnet-1500
python test.py --model_name vggnet-1500
# python test.py --model_name allcolornet
# python test.py --model_name fgbgnet

python evaluate.py --model_name pixelnet-1500 --output_table pixelout-1500.json
python evaluate.py --model_name vggnet-1500 --output_table vggnetout-1500.json
# python evaluate.py --model_name allcolornet --output_table allcolornetout.json
# python evaluate.py --model_name fgbgnet --output_table fgbgnetout.json
