# ADR-004: Multi-Scenario Docker Compose Architecture

| Field | Value |
|-------|-------|
| **Title** | Separate Docker Compose files for dev, test, and production environments |
| **Status** | proposed |
| **Deciders** | DevOps lead, architecture team |
| **Date** | 2025-04-05 |
| **Supersedes** | None |
| **Tags** | docker, compose, devops, deployment, infrastructure |

## Context

The current `docker-compose.yml` provides a single deployment configuration that serves all environments (development, testing, production). This presents several problems:

1. **Development** needs hot-reload, debug ports, verbose logging, and volume-mounted source code
2. **Testing** needs ephemeral containers, mocked LLM calls, coverage reporting, and healthchecks
3. **Production** needs multi-stage builds, non-root users, read-only filesystems, resource limits, logging drivers, and graceful shutdown

A single compose file cannot satisfy all three scenarios without either:
- Maintaining three separate files that drift out of sync
- Using complex conditional logic in a single file

Additionally, the current `Dockerfile` is a single-stage build that:
- Installs optional dependencies (`PyPDF2`, `sentence-transformers`, `rouge-score`) unconditionally
- Lacks a `HEALTHCHECK` instruction
- Has no `.dockerignore`, causing unnecessary image bloat
- Does not optimize for layer caching

## Decision

We will adopt a **multi-file Docker Compose overlay strategy**:

### File Structure

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Base configuration (shared across all environments) |
| `docker-compose.dev.yml` | Development overlay (hot-reload, debug, volumes) |
| `docker-compose.test.yml` | Test overlay (ephemeral, coverage, mocks) |
| `docker-compose.prod.yml` | Production overlay (multi-stage, security, limits) |

### Usage

```bash
# Development
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Testing
docker compose -f docker-compose.yml -f docker-compose.test.yml up --abort-on-container-exit

# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Dockerfile: Multi-Stage Build

We will split the `Dockerfile` into three stages:

1. **`base`** — Python 3.11-slim + core dependencies
2. **`dev`** — Base + optional deps + debugpy + source code
3. **`production`** — Base + minimal deps + gunicorn + compiled code only

### Security Hardening (Production)

- `read_only: true` filesystem with `tmpfs` for `/tmp` and `/app/.cache`
- `security_opt: [no-new-privileges:true]`
- Resource limits: 2 CPU, 4GB RAM
- JSON-file log driver with 10MB × 5 file rotation
- Non-root `appuser`

### Healthchecks

- **Production**: HTTP check on `localhost:7860`
- **Test**: Python import check for `agentic_rag`
- **Dev**: No healthcheck (interferes with debugging)

## Consequences

### Positive
- **Environment parity**: Each environment gets exactly what it needs
- **Smaller production images**: Multi-stage build excludes dev tools and source code
- **Faster dev iteration**: Volume-mounted source with `--cached` flag
- **Clear test isolation**: Ephemeral test containers prevent state leakage
- **Security compliance**: Read-only filesystem + resource limits meet production standards
- **Maintainability**: Shared base config reduces duplication

### Negative
- **Complexity**: Developers must know which compose files to use
- **CI overhead**: Test compose adds another CI pipeline to maintain
- **Compose V2 required**: `docker compose` (V2 syntax) — V1 users need `docker-compose`

### Risks
- **File drift**: If the base compose changes, all overlays must be checked for compatibility
- **Volume permissions**: The non-root user may cause permission issues on host-mounted volumes
- **Healthcheck false positives**: A 200 OK from Gradio does not mean the RAG pipeline is functional (e.g., LLM API could be down)

## Alternatives Considered

### Single Compose File with Profiles
- **Description:** Use Docker Compose profiles (`--profile dev`, `--profile prod`) in a single file
- **Pros:** Simpler file management, fewer files
- **Cons:** Hard to read, mixes concerns, limited conditional logic
- **Why rejected:** Profiles cannot override service build targets or command arguments cleanly enough for our needs.

### Docker BuildKit Bake
- **Description:** Use `docker buildx bake` for multi-target builds
- **Pros:** Excellent for build-time variation
- **Cons:** Does not address runtime compose differences
- **Why rejected:** Complementary to our approach, not a replacement. Could be adopted later.

### Kubernetes Manifests
- **Description:** Use Kubernetes for production deployment
- **Pros:** Scalable, production-grade orchestment
- **Cons:** Overkill for a single-service application, adds massive complexity
- **Why rejected:** Smithy is a single-service Gradio app. Kubernetes adds unnecessary complexity. Compose is sufficient for the current scale.

### Pre-built Docker Images
- **Description:** Publish images to Docker Hub / GHCR and use `image:` instead of `build:`
- **Pros:** No build step for end users, faster startup
- **Cons:** Requires CI image publishing pipeline, trust in registry
- **Why rejected:** Should be adopted once the project reaches stable release. Currently, building from source is preferred for transparency.

## Notes

- The dev compose file optionally includes a local LLM service (`text-generation-inference` serving `Qwen/Qwen2.5-0.5B-Instruct`) via Docker Compose profiles for offline development.
- Production compose uses `gunicorn` instead of Gradio's development server for WSGI compliance and worker management.
- The `.dockerignore` file (see CODE_REVIEW.md Section 2.3) is critical for keeping image sizes small.

## References

- [Docker Compose Override Files](https://docs.docker.com/compose/how-tos/multiple-compose-files/)
- [Dockerfile Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [Docker Security Best Practices](https://docs.docker.com/build/building/best-practices/)
