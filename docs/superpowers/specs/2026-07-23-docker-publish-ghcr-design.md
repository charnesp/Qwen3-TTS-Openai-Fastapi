# Design: Docker image publish via GitHub Actions → GHCR

**Date:** 2026-07-23  
**Status:** Approved for planning

## Goal

Automatically build and publish Docker images to GitHub Container Registry (`ghcr.io`) on every push to `main`, so users can pull pre-built images instead of building locally.

## Decisions

| Topic | Choice |
|---|---|
| Registry | GHCR (`ghcr.io`) |
| Variants | CUDA (`production`), CPU (`cpu-base`), vLLM (`Dockerfile.vllm`) |
| Out of scope | ROCm, Docker Hub, multi-arch, semver tags |
| Trigger | Push to `main` only |
| Tags | `latest` + `sha-<short>` |
| Structure | Single workflow with a build matrix |
| Platform | `linux/amd64` |

## Architecture

File: `.github/workflows/docker-publish.yml`

```
push to main
    → job build-and-push (matrix: cuda | cpu | vllm)
        → checkout
        → setup Buildx
        → login ghcr.io (GITHUB_TOKEN)
        → docker/metadata-action (latest, sha-<short>)
        → docker/build-push-action (push + GHA cache)
```

### Matrix

| Variant | Dockerfile | Target | Image name |
|---|---|---|---|
| `cuda` | `Dockerfile` | `production` | `ghcr.io/<owner>/<repo>` |
| `cpu` | `Dockerfile` | `cpu-base` | `ghcr.io/<owner>/<repo>-cpu` |
| `vllm` | `Dockerfile.vllm` | *(default)* | `ghcr.io/<owner>/<repo>-vllm` |

`<owner>/<repo>` is derived from `github.repository` (lowercase for GHCR).

### Permissions

```yaml
permissions:
  contents: read
  packages: write
```

Auth uses the default `GITHUB_TOKEN`; no extra secrets required for same-repo GHCR packages.

### Caching

BuildKit cache via GitHub Actions cache (`type=gha`), scoped per matrix variant so CUDA/CPU/vLLM caches do not collide.

### Failure isolation

`strategy.fail-fast: false` — one variant failing (e.g. long vLLM build) does not cancel the others.

## Error handling & constraints

- No publish on pull requests (images stay private to successful `main` builds).
- No GPU runtime smoke tests in CI (`ubuntu-latest` has no GPU).
- Runner default timeout (6h) is acceptable for CUDA/vLLM builds.
- Package visibility follows repo/org GHCR defaults; documenting public pull may require a one-time package visibility setting in GitHub UI.

## Documentation

Update README Docker section with pull examples for the three GHCR images. Do not modify Dockerfiles for this change.

## Non-goals

- ROCm image publish
- Multi-architecture manifests (`arm64`, etc.)
- Semver / release tags
- Docker Hub mirror
- Changing `docker-compose.yml` to default to pre-built images (optional follow-up)

## Success criteria

1. Pushing to `main` builds and pushes the three images to GHCR.
2. Each image is tagged `latest` and `sha-<short>`.
3. README documents how to pull and run each variant.
4. A failure in one matrix entry does not block the others.
