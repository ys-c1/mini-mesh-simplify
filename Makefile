.PHONY: test clean

test:
	python3 tests/test_simplify.py

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
