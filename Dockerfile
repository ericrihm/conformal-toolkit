FROM sagemath/sagemath:10.4

WORKDIR /home/sage/conformal-toolkit
COPY . .

# Install both packages in the Sage Python environment
RUN sage -pip install -e ".[dev]" && \
    sage -pip install torch scipy numpy robust-laplacian

CMD ["sage", "-python", "-m", "pytest", "tests/", "-v"]
