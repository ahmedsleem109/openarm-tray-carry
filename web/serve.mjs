// A static server for local development, rooted at the repository so the page
// can fetch the scene and the arm meshes out of third_party/.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const PORT = Number(process.env.PORT ?? 8000);
const TYPES = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".wasm": "application/wasm", ".xml": "text/xml",
  ".png": "image/png", ".stl": "model/stl", ".obj": "text/plain",
  ".mp4": "video/mp4", ".gif": "image/gif", ".css": "text/css",
};

http
  .createServer((req, res) => {
    const url = decodeURIComponent(req.url.split("?")[0]);
    // Mirror GitHub Pages: the site root is the repository root and the page
    // lives at /web/. Serving index.html at "/" instead would resolve its
    // sibling modules against the wrong directory -- which is a 404 here and
    // would have been a 404 in production too.
    if (url === "/") {
      res.writeHead(302, { location: "/web/" }).end();
      return;
    }
    const rel = (url.endsWith("/") ? `${url}index.html` : url).replace(/^\//, "");
    const file = path.join(ROOT, rel);
    if (!file.startsWith(ROOT)) {
      res.writeHead(403).end("no");
      return;
    }
    fs.readFile(file, (err, bytes) => {
      if (err) {
        res.writeHead(404).end(`not found: ${rel}`);
        return;
      }
      res.writeHead(200, {
        "content-type": TYPES[path.extname(file)] ?? "application/octet-stream",
        // MuJoCo's WASM build is happier with these, and they cost nothing.
        "cross-origin-opener-policy": "same-origin",
        "cross-origin-embedder-policy": "require-corp",
      });
      res.end(bytes);
    });
  })
  .listen(PORT, () => console.log(`http://localhost:${PORT}/`));
