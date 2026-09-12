// A browser smoke test: does the page actually load MuJoCo, grasp the tray and
// start carrying, without console errors?
//
// Needs a server running (`npm run serve`) and Playwright's chromium
// (`npx playwright install chromium`). It launches the full browser rather than
// the headless shell on purpose -- the shell runs this WASM about ten times
// slower, which looks like a performance bug in the page and is not one.
//
//     node test-page.mjs [screenshot.png]
import { chromium } from "@playwright/test";
const browser = await chromium.launch({ channel: 'chromium' });
const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
page.on("response", (r) => { if (r.status() >= 400) errors.push(r.status() + " " + r.url()); });
await page.goto("http://localhost:8124/web/", { waitUntil: "load" });
try {
  await page.waitForFunction(
    () => document.getElementById("phase")?.textContent?.includes("carrying"),
    { timeout: 90000 },
  );
  console.log("reached the carrying phase");
} catch {
  console.log("did NOT reach carrying; phase =", await page.textContent("#phase"));
}
await page.waitForTimeout(3000);
console.log("status:", (await page.textContent("#status"))?.slice(0, 120));
console.log("offset:", await page.textContent("#offset"), "tilt:", await page.textContent("#tilt"),
            "grip:", await page.textContent("#grip"), "rate:", await page.textContent("#rate"));
await page.screenshot({ path: process.argv[2] ?? "page.png", fullPage: false });
if (errors.length) console.log("CONSOLE ERRORS:\n " + errors.slice(0, 6).join("\n "));
else console.log("no console errors");
await browser.close();
