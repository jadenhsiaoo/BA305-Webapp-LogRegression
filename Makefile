PYTHON := /usr/bin/python3
VENV := $(shell pwd)/venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install:
	rm -rf venv
	$(PYTHON) -m venv venv
	curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
	$(PY) get-pip.py
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

train:
	$(PY) -c "from target import train_and_save_model; train_and_save_model()"

run:
	FLASK_APP=app.py FLASK_ENV=development $(PY) -m flask run --port 3000
