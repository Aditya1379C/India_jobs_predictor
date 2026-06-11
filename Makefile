# India Tech Jobs — common development commands
# Usage: make <target>

PYTHON  ?= python3
PREDICT  = $(PYTHON) predict.py
PORT    ?= 8080

.PHONY: help install scrape train report serve pipeline predict test clean

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:       ## Install Python dependencies
	$(PYTHON) -m pip install -r requirements.txt

scrape:        ## Scrape fresh job listings (3 pages per keyword)
	$(PREDICT) scrape

train:         ## Train the salary prediction model
	$(PREDICT) train

report:        ## Generate the HTML dashboard report
	$(PREDICT) report

serve:         ## Start the local web dashboard (http://127.0.0.1:$(PORT))
	$(PYTHON) server.py --port $(PORT)

pipeline:      ## Full pipeline: scrape → train → report
	$(PREDICT) scrape
	$(PREDICT) train
	$(PREDICT) report

predict:       ## Interactive salary prediction (prompts for role/city/exp)
	@echo "Example: $(PREDICT) salary --role 'Data Scientist' --city Bangalore --exp 3"
	@$(PREDICT) salary --help

test:          ## Run the unit test suite
	$(PYTHON) -m pytest tests/ -v

clean:         ## Remove cached model files and generated reports
	rm -f models/*.pkl models/*.json
	rm -rf report/*.html
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
