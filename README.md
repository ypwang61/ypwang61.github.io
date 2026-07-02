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
pub.html               Publications
fun.html               Fun (Life + thoughts + Cubi gallery)
miscellaneous.html     Miscellaneous
photos/ pdfs/ project/ evolve/ blog/   Preserved assets and sub-pages
```

## How to update content

- **Add a news item** → append one entry to `_data/news.yml`.
- **Add a paper** → append one entry to `_data/publications.yml`. It shows up on
  `pub.html` automatically; set `featured: true` + `group` + `img` + `tldr` to
  also feature it on the home page's "Main Research".
- **Add a thought** → add one line to `_data/thoughts.yml`.
- **Change the nav / footer** → edit `_includes/nav.html` / `_includes/footer.html`.

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
