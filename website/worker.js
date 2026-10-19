const canonicalHost = 'neurolytics.net';
const homeAliases = new Set(['/index', '/index.html', '/landing', '/landing/', '/landing.html']);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const productionHost = [canonicalHost, `www.${canonicalHost}`, 'nueroanalytics.com', 'www.nueroanalytics.com'].includes(url.hostname);
    const homeAlias = homeAliases.has(url.pathname);
    if (productionHost && (url.hostname !== canonicalHost || url.protocol !== 'https:' || homeAlias)) {
      url.protocol = 'https:';
      url.hostname = canonicalHost;
      if (homeAlias) url.pathname = '/';
      return Response.redirect(url.href, 301);
    }
    // Preserve local previews; production aliases and www consolidate in one hop.
    return env.ASSETS.fetch(request);
  }
};
