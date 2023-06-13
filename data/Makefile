
imagenet:
	kaggle competitions download -c imagenet-object-localization-challenge -f imagenet_object_localization_patched2019.tar.gz
	tar zxvf imagenet_object_localization_patched2019.tar.gz
	cd ILSVRC/Data/CLS-LOC/val && wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
	mkdir imagenet
	mv ILSVRC/Data/CLS-LOC/train imagenet/train
	mv ILSVRC/Data/CLS-LOC/val imagenet/val
	#rm -r ILSVRC

imagenet100:
	mkdir -p imagenet100 imagenet100/train imagenet100/val
	curl https://raw.githubusercontent.com/HobbitLong/CMC/master/imagenet100.txt -o imagenet100/imagenet100.txt
	cat imagenet100/imagenet100.txt | xargs -I % cp -rf imagenet/train/% imagenet100/train/
	cat imagenet100/imagenet100.txt | xargs -I % cp -rf imagenet/val/% imagenet100/val/

