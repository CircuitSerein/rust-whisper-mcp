<!-- Keep PRs focused: one concern per PR. -->

## What & why

<!-- What does this change and what problem does it solve? Link any issue. -->

## How to test

<!-- Steps a reviewer can run. -->

## Checklist

- [ ] `cargo fmt --all -- --check` is clean
- [ ] `cargo clippy --no-default-features --all-targets -- -D warnings` is clean
- [ ] `cargo test --no-default-features` passes
- [ ] No secrets, tokens, or machine-specific absolute paths committed
- [ ] Docs/README updated if behavior or config changed
