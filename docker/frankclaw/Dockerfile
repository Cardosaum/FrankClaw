FROM rust:1.93-bookworm AS builder

WORKDIR /app

# Cache dependency builds: copy manifests first, create stub libs, build deps.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY examples ./examples

RUN cargo build --release -p frankclaw

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /usr/sbin/nologin frankclaw \
    && mkdir -p /var/lib/frankclaw /config \
    && chown frankclaw:frankclaw /var/lib/frankclaw

COPY --from=builder /app/target/release/frankclaw /usr/local/bin/frankclaw

USER frankclaw

ENV FRANKCLAW_CONFIG=/config/frankclaw.json
ENV FRANKCLAW_STATE_DIR=/var/lib/frankclaw

EXPOSE 18789

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://127.0.0.1:18789/health || exit 1

ENTRYPOINT ["frankclaw", "gateway"]
