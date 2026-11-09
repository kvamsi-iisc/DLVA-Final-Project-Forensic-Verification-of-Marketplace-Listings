.PHONY: test notebooks web demo env
test:
	python -m unittest discover -s tests -t .
notebooks:
	python notebooks/make_notebooks.py
web:
	cd web && npm install && npm run build
demo:
	python -m listing_verifier.demo.server --bundle outputs/demo_bundle --offline
env:
	python scripts/env_report.py
