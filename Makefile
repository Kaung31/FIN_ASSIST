.PHONY: install lint test index run ui
install: ; pip install -e ".[dev]"
lint:    ; ruff check src tests && ruff format --check src tests
test:    ; pytest -q
index:   ; python scripts/create_index.py
run:     ; uvicorn finassist.api.main:app --reload
ui:      ; streamlit run ui/streamlit_app.py
