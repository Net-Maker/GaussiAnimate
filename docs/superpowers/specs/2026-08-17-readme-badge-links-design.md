# README Badge Links

## Scope

Replace the four plain-text links below the README subtitle with four
Shields.io `for-the-badge` links. No other README content changes.

## Design

- Project Page: blue badge with the Google Chrome logo.
- Paper: red badge with the arXiv logo and paper identifier.
- Experiments: teal badge with a documentation logo, linking to
  `experiments/README.md`.
- License: green badge with the Open Source Initiative logo, linking to
  `LICENSE`.

Use plain Markdown image links so GitHub handles horizontal layout and wraps
the badges naturally on narrow screens. Badge labels must remain readable if
images are unavailable through descriptive alt text.

## Verification

- Confirm the two local targets exist.
- Confirm the two external destinations and four badge images respond.
- Run `git diff --check`.
