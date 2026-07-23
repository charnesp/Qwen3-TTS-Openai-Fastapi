# Docker Publish to GHCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On every push to `main`, build and publish CUDA, CPU, and vLLM Docker images to GitHub Container Registry (`ghcr.io`).

**Architecture:** One GitHub Actions workflow with a three-entry matrix. Each entry builds a specific Dockerfile/target, tags the image as `latest` and `sha-<short>`, and pushes to GHCR using `GITHUB_TOKEN`. README documents pull commands for the three images.

**Tech Stack:** GitHub Actions, Docker Buildx, `docker/build-push-action`, `docker/metadata-action`, `docker/login-action`, GHCR

## Global Constraints

- Registry: `ghcr.io` only (no Docker Hub)
- Variants: CUDA (`Dockerfile` target `production`), CPU (`Dockerfile` target `cpu-base`), vLLM (`Dockerfile.vllm`)
- No ROCm publish
- Trigger: push to `main` only
- Tags: `latest` + `sha-<short>` (via `type=sha,prefix=sha-`)
- Platform: `linux/amd64`
- `fail-fast: false`
- Cache: BuildKit GHA cache scoped per variant
- Do not modify Dockerfiles
- Auth: default `GITHUB_TOKEN` with `packages: write`

## File Structure

| File | Responsibility |
|---|---|
| `.github/workflows/docker-publish.yml` | Build matrix + login + metadata + build/push to GHCR |
| `README.md` | Document pulling pre-built GHCR images |

---

### Task 1: Add docker-publish GitHub Actions workflow

**Files:**
- Create: `.github/workflows/docker-publish.yml`

**Interfaces:**
- Consumes: existing `Dockerfile` (`production`, `cpu-base`), `Dockerfile.vllm`
- Produces: three GHCR packages on successful `main` push:
  - `ghcr.io/<owner>/<repo>:latest` / `:sha-<short>`
  - `ghcr.io/<owner>/<repo>-cpu:latest` / `:sha-<short>`
  - `ghcr.io/<owner>/<repo>-vllm:latest` / `:sha-<short>`

- [ ] **Step 1: Create the workflow file**

Create `.github/workflows/docker-publish.yml` with exactly this content:

```yaml
name: Docker publish

on:
  push:
    branches: [main]

permissions:
  contents: read
  packages: write

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        include:
          - variant: cuda
            dockerfile: Dockerfile
            target: production
            image_suffix: ""
          - variant: cpu
            dockerfile: Dockerfile
            target: cpu-base
            image_suffix: "-cpu"
          - variant: vllm
            dockerfile: Dockerfile.vllm
            target: ""
            image_suffix: "-vllm"

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract Docker metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}${{ matrix.image_suffix }}
          tags: |
            type=raw,value=latest
            type=sha,prefix=sha-

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ${{ matrix.dockerfile }}
          target: ${{ matrix.target }}
          platforms: linux/amd64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha,scope=${{ matrix.variant }}
          cache-to: type=gha,mode=max,scope=${{ matrix.variant }}
```

Notes for the implementer:
- `docker/metadata-action` lowercases the image name for GHCR.
- Empty `target` for vLLM means Buildx builds the final stage of `Dockerfile.vllm` (single-stage file).
- Do not add `workflow_dispatch`, PR triggers, or ROCm entries.

- [ ] **Step 2: Validate workflow YAML syntax**

Run:

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/docker-publish.yml')); print('OK')"
```

If `PyYAML` is missing:

```bash
python3 -c "from pathlib import Path; p=Path('.github/workflows/docker-publish.yml'); print('exists', p.exists(), 'bytes', p.stat().st_size)"
```

Expected: `OK` (or file exists with non-zero size). Optionally, if `actionlint` is installed:

```bash
actionlint .github/workflows/docker-publish.yml
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/docker-publish.yml
git commit -m "$(cat <<'EOF'
ci: publish CUDA, CPU, and vLLM images to GHCR

Build and push matrix images on main with latest and sha tags.
EOF
)"
```

---

### Task 2: Document GHCR pull commands in README

**Files:**
- Modify: `README.md` (Docker section, immediately after the `## Docker` heading and before `### NVIDIA GPU`)

**Interfaces:**
- Consumes: image naming from Task 1 (`ghcr.io/<owner>/<repo>`, `-cpu`, `-vllm`)
- Produces: user-facing pull/run examples for the three published images

- [ ] **Step 1: Insert pre-built images subsection**

In `README.md`, find:

```markdown
## Docker

### NVIDIA GPU
```

Replace with:

```markdown
## Docker

### Pre-built images (GHCR)

Images are published to GitHub Container Registry on every push to `main`:

| Variant | Image |
|---|---|
| NVIDIA CUDA | `ghcr.io/charnesp/qwen3-tts-openai-fastapi:latest` |
| CPU | `ghcr.io/charnesp/qwen3-tts-openai-fastapi-cpu:latest` |
| vLLM | `ghcr.io/charnesp/qwen3-tts-openai-fastapi-vllm:latest` |

Also tagged as `sha-<short>` for pin-to-commit pulls.

```bash
# CUDA (GPU)
docker pull ghcr.io/charnesp/qwen3-tts-openai-fastapi:latest
docker run --gpus all -p 8880:8880 ghcr.io/charnesp/qwen3-tts-openai-fastapi:latest

# CPU
docker pull ghcr.io/charnesp/qwen3-tts-openai-fastapi-cpu:latest
docker run -p 8880:8880 ghcr.io/charnesp/qwen3-tts-openai-fastapi-cpu:latest

# vLLM
docker pull ghcr.io/charnesp/qwen3-tts-openai-fastapi-vllm:latest
docker run --gpus all -p 8880:8880 ghcr.io/charnesp/qwen3-tts-openai-fastapi-vllm:latest
```

If pulls fail with unauthorized, set the package visibility to Public under the repo's GitHub Packages settings (one-time).

### NVIDIA GPU
```

Use the lowercase GHCR form of `charnesp/Qwen3-TTS-Openai-Fastapi` → `charnesp/qwen3-tts-openai-fastapi` (metadata-action sanitization).

- [ ] **Step 2: Verify README still mentions local compose builds**

Confirm the following subsections still exist unchanged after the new block:
- `### NVIDIA GPU` with `docker compose up --build qwen3-tts-gpu`
- `### vLLM` with `--profile vllm`
- `### CPU` with `--profile cpu`
- `### AMD ROCm`

Run:

```bash
rg -n "## Docker|Pre-built images|### NVIDIA GPU|### vLLM|### CPU|### AMD ROCm" README.md
```

Expected: all headings present, with `Pre-built images` appearing before `NVIDIA GPU`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: document GHCR pre-built Docker image pulls

Add CUDA, CPU, and vLLM pull/run examples for packages published from main.
EOF
)"
```

---

### Task 3: Smoke-check workflow wiring against the repo

**Files:**
- Test: no new test file; verify local references only

**Interfaces:**
- Consumes: Task 1 workflow matrix + existing Dockerfiles
- Produces: confirmation that every matrix entry points at a real file/target

- [ ] **Step 1: Confirm Dockerfiles and targets exist**

Run:

```bash
test -f Dockerfile && test -f Dockerfile.vllm && \
rg -n "^FROM .* AS (production|cpu-base)" Dockerfile && \
echo "Dockerfiles OK"
```

Expected: matches for `production` and `cpu-base`, and `Dockerfiles OK`.

- [ ] **Step 2: Confirm workflow references match**

Run:

```bash
rg -n "dockerfile:|target:|image_suffix:|branches:|packages: write|type=sha,prefix=sha-|scope=" .github/workflows/docker-publish.yml
```

Expected:
- `branches: [main]`
- `packages: write`
- matrix entries for `Dockerfile`/`production`, `Dockerfile`/`cpu-base`, `Dockerfile.vllm`
- `type=sha,prefix=sha-`
- cache `scope=${{ matrix.variant }}`

- [ ] **Step 3: Final commit only if Step 1–2 required fixes**

If any mismatch was found and fixed, commit with:

```bash
git add .github/workflows/docker-publish.yml README.md
git commit -m "$(cat <<'EOF'
fix: align docker publish workflow with Dockerfile targets
EOF
)"
```

If nothing needed fixing, skip the commit.

---

## Spec coverage checklist

| Spec requirement | Task |
|---|---|
| GHCR registry | Task 1 |
| CUDA + CPU + vLLM variants | Task 1 |
| Push to `main` only | Task 1 |
| Tags `latest` + `sha-<short>` | Task 1 |
| Matrix workflow, `fail-fast: false` | Task 1 |
| GHA cache per variant | Task 1 |
| `linux/amd64` | Task 1 |
| README pull docs | Task 2 |
| No Dockerfile changes | (none modified) |
| No ROCm / multi-arch / semver | (omitted) |

## Execution notes

- First run on `main` after merge will create GHCR packages; visibility may default to private — README already covers the one-time Public toggle.
- CUDA and vLLM builds can take a long time on GitHub-hosted runners; that is expected.
- Do not add GPU smoke tests; runners have no GPU.
