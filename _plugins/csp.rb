# frozen_string_literal: true

# Jekyll plugin to patch the Content-Security-Policy meta tag injected
# by al_folio_core's head.liquid.
#
# The default CSP blocks tikzjax Web Workers (needs worker-src blob:)
# and WebAssembly compilation (needs unsafe-eval in script-src).
# This plugin replaces the CSP meta tag in rendered HTML pages.

CSP_RE = %r{<meta\s+http-equiv="Content-Security-Policy"\s+content="([\s\S]*?)"\s*/?>}.freeze
NEW_CSP = '<meta http-equiv="Content-Security-Policy" content="default-src \'self\'; script-src \'self\' \'unsafe-inline\' \'unsafe-eval\' https:; worker-src \'self\' blob:; style-src \'self\' \'unsafe-inline\' https:; img-src \'self\' data: https:; font-src \'self\' data: https:; media-src \'self\' https:; frame-src \'self\' https:; connect-src \'self\' https:;">'

def patch_csp(output)
  output.gsub(CSP_RE, NEW_CSP)
end

Jekyll::Hooks.register :pages, :post_render do |page|
  page.output = patch_csp(page.output) if page.output.respond_to?(:gsub)
end

Jekyll::Hooks.register :documents, :post_render do |doc|
  doc.output = patch_csp(doc.output) if doc.output.respond_to?(:gsub)
end
