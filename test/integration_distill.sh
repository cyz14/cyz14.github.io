#!/usr/bin/env bash
set -euo pipefail

tmp_dir="$(mktemp -d)"
tmp_override="${tmp_dir}/distill-override.yml"
tmp_site="${tmp_dir}/site"
tmp_source="${tmp_dir}/source"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

mkdir -p "${tmp_source}/_posts"
cp -R _data "${tmp_source}/_data"
cat >"${tmp_source}/_posts/2000-01-01-distill.md" <<'MARKDOWN'
---
layout: distill
title: Distill integration fixture
description: Cross-plugin rendering fixture
permalink: /test/distill/
giscus_comments: true
mermaid:
  enabled: true
tikzjax: true
---
Distill integration fixture.
MARKDOWN

cat >"${tmp_override}" <<'YAML'
jekyll_get_json: []
giscus:
  repo: alshedivat/al-folio
  repo_id: R_kgDOExample
  category: Comments
  category_id: DIC_kwDOExample
YAML

bundle exec jekyll build --source "${tmp_source}" --config "_config.yml,${tmp_override}" -d "${tmp_site}" >/dev/null

distill_page="${tmp_site}/test/distill/index.html"

if [ ! -f "${distill_page}" ]; then
  echo "distill page was not generated at ${distill_page}" >&2
  exit 1
fi

grep -q 'd-front-matter' "${distill_page}"
grep -q '/assets/js/distillpub/template.v2.js' "${distill_page}"
grep -q '/assets/js/distillpub/transforms.v2.js' "${distill_page}"
grep -q '/assets/js/distillpub/overrides.js' "${distill_page}"
grep -q '/assets/al_charts/js/mermaid-setup.js' "${distill_page}"
for asset in css js; do
  resource_url="$(bundle exec ruby -rpsych -e 'puts Psych.unsafe_load_file("_config.yml").fetch("third_party_libraries").fetch("tikzjax").fetch("url").fetch(ARGV.fetch(0))' "${asset}")"
  if [ -z "${resource_url}" ] || ! grep -Fq "${resource_url}" "${distill_page}"; then
    echo "missing configured TikZJax ${asset} resource in ${distill_page}: ${resource_url}" >&2
    exit 1
  fi
done
grep -q 'id="giscus_thread"' "${distill_page}"
transforms_runtime="${tmp_site}/assets/js/distillpub/transforms.v2.js"
distill_runtime="$(PATH="$HOME/.rbenv/shims:$PATH" bundle exec ruby -e 'spec = Gem.loaded_specs["al_folio_distill"]; puts(spec ? File.join(spec.full_gem_path, "assets/js/distillpub/transforms.v2.js") : "")')"
if [ -f "${distill_runtime}" ]; then
  # Prefer the packaged gem runtime for deterministic parity checks.
  transforms_runtime="${distill_runtime}"
elif [ ! -f "${transforms_runtime}" ]; then
  echo "distill transforms runtime missing at ${transforms_runtime} (and not found in installed al_folio_distill gem)" >&2
  exit 1
fi

expected_transforms_hash="70e3f488e23ec379d33a10a60311ec60b570b3b2d5f1823e9159f661c315184e"
actual_transforms_hash="$(ruby -rdigest -e 'print Digest::SHA256.file(ARGV[0]).hexdigest' "${transforms_runtime}")"
if [ "${actual_transforms_hash}" != "${expected_transforms_hash}" ]; then
  echo "unexpected distill transforms runtime hash: ${actual_transforms_hash}" >&2
  exit 1
fi

echo "distill integration checks passed"
