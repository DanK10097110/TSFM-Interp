# Demo site (GitHub Pages)

`index.html` is the landing page; `build_site.sh` assembles it together with the
example reports and the paper PDFs at deploy time (`.github/workflows/pages.yml`).
Nothing large is committed under a `docs/` folder. Files missing on the branch
(for example the PDFs) are skipped and their entries dropped from the page.

Preview locally: `bash site/build_site.sh site_out` then open `site_out/index.html`.

## One manual step

In the GitHub repo: **Settings -> Pages -> Build and deployment -> Source:
GitHub Actions**. After the next push to `main` (or a manual run of the `pages`
workflow) the site is at `https://<user>.github.io/<repo>/`, here
`https://danK10097110.github.io/TSFM-Interp/` (GitHub lowercases the user name).
