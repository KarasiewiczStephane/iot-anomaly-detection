.PHONY: install test lint lint-fix clean run docker dashboard generate-data

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage coverage.xml htmlcov

run:
	python -m src.main

dashboard:
	streamlit run src/dashboard/app.py

docker:
	docker build -t $(shell basename $(CURDIR)) .

generate-data:
	python -c "from src.data.generator import IoTDataGenerator; g = IoTDataGenerator(); df = g.generate(days=30); g.save_to_csv(df, 'data/generated/sensor_data.csv'); print(f'Generated {len(df)} rows')"
