FROM ann-benchmarks

# https://github.com/pgvector/pgvector/blob/master/Dockerfile

RUN git clone https://github.com/pgvector/pgvector /tmp/pgvector

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
RUN apt-get update && apt-get install -y --no-install-recommends build-essential postgresql postgresql-server-dev-all
RUN sh -c 'echo "local all all trust" > /etc/postgresql/14/main/pg_hba.conf'
RUN cd /tmp/pgvector && \
	make clean && \
	make OPTFLAGS="-march=native -mprefer-vector-width=512" && \
	make install

USER postgres
RUN service postgresql start && \
    psql -c "CREATE USER ann WITH ENCRYPTED PASSWORD 'ann'" && \
    psql -c "CREATE DATABASE ann" && \
    psql -c "GRANT ALL PRIVILEGES ON DATABASE ann TO ann" && \
    psql -d ann -c "CREATE EXTENSION vector" && \
    psql -c "ALTER USER ann SET maintenance_work_mem = '16GB'"
USER root

RUN pip install psycopg[binary] pgvector

# Create a script to modify postgresql.conf during container initialization
RUN echo '#!/bin/bash' > /docker-entrypoint-initdb.d/modify-config.sh && \
    echo "sed -i 's/#shared_buffers = 128MB/shared_buffers = 64GB/' \$PGDATA/postgresql.conf" >> /docker-entrypoint-initdb.d/modify-config.sh && \
    echo "sed -i 's/#effective_cache_size = 4GB/effective_cache_size = 128GB/' \$PGDATA/postgresql.conf" >> /docker-entrypoint-initdb.d/modify-config.sh && \
    echo "sed -i 's/#work_mem = 4MB/work_mem = 4GB/' \$PGDATA/postgresql.conf" >> /docker-entrypoint-initdb.d/modify-config.sh && \
    echo "sed -i 's/#maintenance_work_mem = 64MB/maintenance_work_mem = 32GB/' \$PGDATA/postgresql.conf" >> /docker-entrypoint-initdb.d/modify-config.sh && \
    echo "sed -i 's/#max_wal_size = 1GB/max_wal_size = 10GB/' \$PGDATA/postgresql.conf" >> /docker-entrypoint-initdb.d/modify-config.sh && \
    echo "sed -i 's/#wal_compression = off/wal_compression = zstd/' \$PGDATA/postgresql.conf" >> /docker-entrypoint-initdb.d/modify-config.sh && \
    echo "sed -i 's/#checkpoint_timeout = 30min/checkpoint_timeout = 2h/' \$PGDATA/postgresql.conf" >> /docker-entrypoint-initdb.d/modify-config.sh && \
    echo "sed -i 's/#jit = on/jit = off/' \$PGDATA/postgresql.conf" >> /docker-entrypoint-initdb.d/modify-config.sh && \
    chmod +x /docker-entrypoint-initdb.d/modify-config.sh
