DOCKER_IMAGE=yutaros/scalable-decorrelation-ssl:1.0

#####################################

solo-init:
	git submodule update --init --recursive

solo-main:
	cp src/main/main_pretrain.py solo-learn/main_pretrain.py
	cd solo-learn/solo/methods && \
		ln -fs ../../../src/main/methods/__init__.py __init__.py && \
		ln -fs ../../../src/main/methods/svicreg.py svicreg.py && \
		ln -fs ../../../src/main/methods/sbarlow_twins.py sbarlow_twins.py
	cd solo-learn/solo/losses && \
		ln -fs ../../../src/main/losses/__init__.py __init__.py && \
		ln -fs ../../../src/main/losses/util.py util.py && \
		ln -fs ../../../src/main/losses/svicreg.py svicreg.py && \
		ln -fs ../../../src/main/losses/sbarlow.py sbarlow.py

solo-profile:
	cp -rf solo-learn solo-learn-profile
	cp src/profile/main_pretrain_mem_profile.py solo-learn-profile/main_pretrain_mem_profile.py
	cd solo-learn-profile/solo/methods && \
		ln -fs ../../../src/profile/methods/base.py base.py && \
		ln -fs ../../../src/profile/methods/barlow_twins.py barlow_twins.py && \
		ln -fs ../../../src/profile/methods/sbarlow_twins.py sbarlow_twins.py && \
		ln -fs ../../../src/profile/methods/vicreg.py vicreg.py && \
		ln -fs ../../../src/profile/methods/svicreg.py svicreg.py

#####################################

docker-build:
	cd docker && docker build -t ${DOCKER_IMAGE} .

docker-run:
	docker run --gpus all -it --rm --shm-size 100G -e WANDB_API_KEY -e WANDB_ENTITY -v $(PWD):/workspace -w /workspace $(DOCKER_IMAGE)

#####################################

download-data:
	make -C data
