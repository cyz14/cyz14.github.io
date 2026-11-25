#!/usr/bin/env bash
set -euo pipefail

tmp_dir="$(mktemp -d)"
tmp_override="${tmp_dir}/comments-test-override.yml"
tmp_site="${tmp_dir}/site"
tmp_source="${tmp_dir}/source"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

# Test fixtures are independent of a site's published example posts.
mkdir -p "${tmp_source}/_posts"
cp -R _data "${tmp_source}/_data"
cat >"${tmp_source}/_posts/2000-01-01-giscus.md" <<'MARKDOWN'
---
layout: post
title: Giscus integration fixture
permalink: /test/giscus/
giscus_comments: true
related_posts: false
---
Giscus comments fixture.
MARKDOWN
cat >"${tmp_source}/_posts/2000-01-02-disqus.md" <<'MARKDOWN'
---
layout: post
title: Disqus integration fixture
permalink: /test/disqus/
disqus_comments: true
related_posts: false
---
Disqus comments fixture.
MARKDOWN

cat >"${tmp_override}" <<'YAML'
jekyll_get_json: []
disqus_shortname: al-folio
giscus:
  repo: alshedivat/al-folio
  repo_id: R_kgDOExample
  category: Comments
  category_id: DIC_kwDOExample
YAML

bundle exec jekyll build --source "${tmp_source}" --config "_config.yml,${tmp_override}" -d "${tmp_site}" >/dev/null

giscus_page="${tmp_site}/test/giscus/index.html"
disqus_page="${tmp_site}/test/disqus/index.html"

grep -q 'https://giscus.app/client.js' "${giscus_page}"
if grep -q 'giscus comments misconfigured' "${giscus_page}"; then
  echo "unexpected giscus misconfiguration warning in ${giscus_page}" >&2
  exit 1
fi

grep -q 'id="disqus_thread"' "${disqus_page}"
grep -q '.disqus.com/embed.js' "${disqus_page}"

echo "comments integration checks passed"
