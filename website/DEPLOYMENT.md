# NVAP website deployment

Cloudflare Worker: `nvap-landing` in `Admin@protoron.com's Account`.

Production hostnames: `neurolytics.net` and `www.neurolytics.net`.
The previous `nueroanalytics.com` and `www.nueroanalytics.com` routes remain
attached to serve permanent redirects to the new domain.
Cloudflare URL: https://nvap-landing.restless-cloud-e25f.workers.dev

From this directory:

```powershell
npm ci
npm run check:deploy
npm run deploy
```

Wrangler uses the existing local Cloudflare login. No credentials are stored in
this repository. `wrangler.jsonc` pins the intended account and hostname routes.

The build copies only the landing page and its referenced assets into `public/`;
the production root serves the landing page directly. Analysis inputs, unused
screenshots, app sources and deployment tooling are excluded.

Namecheap delegates DNS to `celeste.ns.cloudflare.com` and
`eugene.ns.cloudflare.com`. Cloudflare uses proxied apex and www records with
Worker routes covering all paths. Static assets are served by the Worker;
the imported parking origin is not used for site requests. Imported MX and SPF
records are preserved. Cloudflare manages the universal HTTPS certificate.

The canonical homepage is `https://neurolytics.net/`. The Worker permanently
redirects www and legacy landing/index URLs to that host in one hop, preserving
query strings. Other paths retain their path when redirecting www. Static
content still comes from the ASSETS binding; no parking-origin requests occur.
`robots.txt` allows crawling and points to the single-page XML sitemap.
The HTML includes canonical, social-preview, and WebSite/WebPage/SoftwareApplication
JSON-LD metadata. There are no invented reviews, ratings, or prices; the software
schema describes the product without claiming Google app rich-result eligibility.

For indexing visibility, add this domain to Google Search Console and submit
`https://neurolytics.net/sitemap.xml`. Search Console verification and submission
have not been performed by the deployment process. Rankings and indexing are not
guaranteed by metadata or sitemap availability.
