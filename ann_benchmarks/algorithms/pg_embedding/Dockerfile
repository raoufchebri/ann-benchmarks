FROM ann-benchmarks

RUN git clone https://github.com/neondatabase/pg_embedding /tmp/pg_embedding

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
RUN apt-get update && apt-get install -y --no-install-recommends build-essential postgresql postgresql-server-dev-all
RUN sh -c 'echo "local all all trust" > /etc/postgresql/14/main/pg_hba.conf'
RUN cd /tmp/pg_embedding && \
	make clean && \
	make OPTFLAGS="-march=native -mprefer-vector-width=512" && \
	make install

# Modify shared_buffers in postgresql.conf to 16GB
# RUN sed -i 's/#shared_buffers = 128MB/shared_buffers = 16GB/' /etc/postgresql/14/main/postgresql.conf

USER postgres
RUN service postgresql start && \
    psql -c "CREATE USER ann WITH ENCRYPTED PASSWORD 'ann'" && \
    psql -c "CREATE DATABASE ann" && \
    psql -c "GRANT ALL PRIVILEGES ON DATABASE ann TO ann" && \
    psql -d ann -c "CREATE EXTENSION embedding" && \
    psql -d ann -c "CREATE EXTENSION pg_prewarm"
USER root

RUN pip install psycopg[binary]

# Create a script to modify postgresql.conf during container initialization
RUN echo '#!/bin/bash' > /docker-entrypoint-initdb.d/modify-config.sh && \
    echo "sed -i 's/#shared_buffers = 128MB/shared_buffers = 128GB/' \$PGDATA/postgresql.conf" >> /docker-entrypoint-initdb.d/modify-config.sh && \
    echo "sed -i 's/#effective_cache_size = 4GB/effective_cache_size = 256GB/' \$PGDATA/postgresql.conf" >> /docker-entrypoint-initdb.d/modify-config.sh && \
    echo "sed -i 's/#work_mem = 4MB/work_mem = 4GB/' \$PGDATA/postgresql.conf" >> /docker-entrypoint-initdb.d/modify-config.sh && \
    echo "sed -i 's/#maintenance_work_mem = 64MB/maintenance_work_mem = 64GB/' \$PGDATA/postgresql.conf" >> /docker-entrypoint-initdb.d/modify-config.sh && \
    echo "sed -i 's/#max_wal_size = 1GB/max_wal_size = 20GB/' \$PGDATA/postgresql.conf" >> /docker-entrypoint-initdb.d/modify-config.sh && \
    echo "sed -i 's/#wal_compression = off/wal_compression = zstd/' \$PGDATA/postgresql.conf" >> /docker-entrypoint-initdb.d/modify-config.sh && \
    echo "sed -i 's/#checkpoint_timeout = 30min/checkpoint_timeout = 2h/' \$PGDATA/postgresql.conf" >> /docker-entrypoint-initdb.d/modify-config.sh && \
    echo "sed -i 's/#jit = on/jit = off/' \$PGDATA/postgresql.conf" >> /docker-entrypoint-initdb.d/modify-config.sh && \
    chmod +x /docker-entrypoint-initdb.d/modify-config.sh
