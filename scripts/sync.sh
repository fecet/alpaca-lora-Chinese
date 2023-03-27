rsync -avrP --exclude ".*" --exclude "__pycache__" --exclude "data" --delete "/home/rok/Nutstore Files/codes/alpaca-lora-Chinese" node1:~/
rsync -avrP --exclude ".*" --exclude "__pycache__" --exclude "data" --delete "/home/rok/Nutstore Files/codes/alpaca-lora-Chinese" node2:~/
echo "syn code end"
