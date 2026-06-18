.PHONY: run deploy build clean

run:
	@echo "Starting development server..."
	docker compose up --build

deploy:
	@echo "Deploying application..."
	$(MAKE) clean
	$(MAKE) build

build:
	@test -f .env || (echo "Missing .env file. Please create one based on .env.example." && exit 1)
	@echo "Building Docker images..."
	docker compose up --build

clean:
	@echo "Removing build artifacts and caches..."
	rm -rf .venv/ __pycache__/ build/ dist/ *.egg-info/ docker-compose.override.yml
	@echo "Project cleaned."
