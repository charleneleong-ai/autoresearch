# Contributing to autoresearch

## Setup

```bash
mise run init     # creates .venv, installs deps, registers pre-commit hooks
mise run test     # pytest
mise run lint     # ruff
```

`mise run init` registers two pre-commit hook types:
- **`pre-commit`** — runs `ruff-check --fix`, `ruff-format`, and basic file hygiene (whitespace, EOF, large-file guard) before every commit
- **`commit-msg`** — validates conventional-commit format (catches `release.yml` blocker before CI)

## Branching + PRs

- Feature branches off `main`: `feat/<topic>`, `fix/<topic>`, `chore/<topic>`
- One concern per PR
- PR titles and commit messages **must** follow [conventional commits](https://www.conventionalcommits.org/) — release automation depends on it (see "How releases happen" below)

## Conventional commit shape

```
<type>(<optional scope>): <subject>

<optional body — what + why>

<optional footer — BREAKING CHANGE: <description>>
```

Types that affect releases:

| type | semver bump | example |
|---|---|---|
| `feat:` | MINOR | `feat(daemons): add --poll-s envvar` |
| `fix:` | PATCH | `fix(render): handle empty results.jsonl` |
| `BREAKING CHANGE:` (or `feat!:`) | MAJOR (or MINOR while alpha) | `feat!: rename --tag flag to --task` |

Types that **don't** trigger a release:

`chore:` · `docs:` · `refactor:` · `test:` · `style:` · `perf:` · `build:` · `ci:` · `revert:`

If a PR contains only no-bump types since the last tag, `release.yml` exits as a no-op. That's fine — releases happen when there's user-visible change to ship.

## How releases happen

`.github/workflows/release.yml` auto-fires on every push to `main`:

1. **Skip-on-bump** — if the head commit starts with `bump:` (the workflow's own commit), exit early to avoid a release loop
2. `cz bump --dry-run` decides the next version from conventional commits since the last tag
3. If a bump is warranted: `cz bump --yes` updates `pyproject.toml` + `__init__.py` + prepends to `CHANGELOG.md`, commits, tags, pushes back to `main`
4. Builds wheel + sdist; runs `pytest` against the built wheel
5. Creates GitHub Release with auto-generated notes + wheel/sdist attached

You don't need to touch versions, tag, or write release notes manually.

## When CI breaks

Pre-commit (`mise run init` registers it) catches the same checks CI runs, so failures here mean either:
- pre-commit was bypassed (`git commit --no-verify`)
- a new lint rule was added without local sync (`mise run sync` to refresh)
- a real packaging or test regression

Recovery paths:

### Path A — autofixable

```bash
ruff check src/ tests/ --fix --unsafe-fixes
git add -u && git commit -m "fix(lint): apply ruff autofixes"
git push
```

CI does **not** auto-push fixes back to the PR — too noisy in the commit history. Fix locally, push, retry.

### Path B — config workaround (false positives like `B008` for typer)

Edit `pyproject.toml`'s `[tool.ruff.lint].ignore` list. Document why in a comment so it doesn't look arbitrary.

### Path C — Claude Code triage

If the failure is non-obvious, paste the failing CI run URL into Claude Code and ask:

> Run `gh run view <run-id> --repo charleneleong-ai/autoresearch --log-failed` to read the failure, then propose a fix. If it's a deterministic lint/format issue, apply it; if it's a logic regression, surface the root cause and ask before patching.

This is what worked for the v0.1.0 prep — Claude read the ruff log, applied autofixes for `UP045`, configured `B008` ignore in pyproject, and broke up `E501` long lines manually. ~5min round-trip, no manual log-reading required.

A future PR may wire this into a proper named Claude skill (`/ci-triage`); for now the prompt above is the recipe.

## Project conventions

- **No emojis in code or commit messages** unless explicitly part of user-visible output (e.g. rich.print markup like `[green]✓[/green]`)
- **No agent-vendor brand names** in docstrings/docs — use "coding-agent" generically. The package is intentionally tool-agnostic
- **Daemon scripts use `python -u` (or `flush=True`)** so live logs aren't buffered when run under `nohup`
