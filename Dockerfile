FROM sagemath/sagemath:10.4
WORKDIR /app
COPY . .
RUN sage -pip install -e ".[dev]"
CMD ["sage", "-python", "-m", "pytest", "tests/", "-v"]
