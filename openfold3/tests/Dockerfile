# Copy Openfold3 folder and build dependencies needed for testing
FROM openfoldconsortium/openfold3:stable

COPY openfold3 /opt/openfold3
COPY environments/development.txt /opt/openfold3/development.txt

WORKDIR /opt/openfold3
RUN pip install -r development.txt
RUN pip install --force-reinstall --no-deps .