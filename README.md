# ypwang61.github.io

Personal academic homepage of Yiping Wang, built as a **minimal, no-plugin
Jekyll** site (GitHub Pages native — nothing to build locally).

## Structure

```
_config.yml            Jekyll config (no theme / no plugins)
_layouts/base.html     Shared page shell
_includes/             head / nav / footer — written once, reused everywhere
_data/                 All content lives here (edit these to update the site)
  news.yml             Key News items
  publications.yml     All papers (single source for both pages)
  thoughts.yml         "Favorite or Random Thoughts" on the Fun page
assets/css/style.css   The single stylesheet
index.html             Home       (layout: base)
pub.html               Publications & Projects
fun.html               Fun (Life + thoughts + Cubi gallery)
miscellaneous.html     Miscellaneous
photos/ pdfs/ project/ evolve/ blog/   Preserved assets and sub-pages
```

## How to update content

- **Add a news item** → append one entry to `_data/news.yml`. (Key News is
  currently hidden on the home page; the data is kept — re-enable by removing the
  `{% comment %}` wrapper in `index.html`.)
- **Add a paper / project** → append one entry to `_data/publications.yml`; it
  shows up on `pub.html` (Publications & Projects) automatically. Add an optional
  `tldr` for a one-line description (used e.g. by the ScaleAutoResearch project).
- **Add a thought** → append an entry to `_data/thoughts.yml` (`body:` + optional `date:`).
- **Change the nav / footer** → edit `_includes/nav.html` / `_includes/footer.html`.
- **Theme** → light/dark is driven by CSS variables in `assets/css/style.css`
  (`:root` and `:root[data-theme="dark"]`). A tiny inline script in
  `_includes/head.html` applies the saved/OS theme before first paint; the toggle
  button in `_includes/nav.html` is wired by a small script in `_includes/footer.html`.

Entries may contain inline HTML (e.g. `<a>`, `<b>`).

## Preview locally (optional)

GitHub Pages builds automatically on push. To preview locally:

```
gem install jekyll bundler   # once, if Jekyll is not installed
jekyll serve                 # then open http://localhost:4000
```

## Content check

`check_site.py` verifies the Jekyll structure and that every required piece of
content + referenced local asset is present:

```
python check_site.py .       # expects "SITE CHECK: PASS"
```
