install:
	pip install -r requirements.txt
download:
	python -m src.data.download
prepare:
	python -m src.data.prepare --data-dir data/mitdb --output data/processed
train:
	python -m src.train --config configs/default.yaml
dashboard:
	streamlit run app.py
test:
	pytest -q
