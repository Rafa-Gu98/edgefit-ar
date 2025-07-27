install:
	pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
	cd edge_gateway/rust_engine && python build.py

train:
	python training/train_model.py

run:
	python edge_gateway/main.py &
	python ar_interface/web_simulator.py

clean:
	rm -rf training/outputs/*