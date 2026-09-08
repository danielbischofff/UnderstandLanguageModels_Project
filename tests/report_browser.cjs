// Real browser regression check, invoked by test_report.py when Playwright is available.
const assert = require('node:assert/strict');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright');

(async () => {
  const browser = await chromium.launch(process.env.BROWSER_EXECUTABLE
    ? {executablePath: process.env.BROWSER_EXECUTABLE, headless: true}
    : {channel: 'chrome', headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1366, height: 768}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(pathToFileURL(process.argv[2]).href);
    assert.equal(await page.locator('.layer').count(), 63);
    const grid = await page.locator('#layer-grid').boundingBox();
    assert(grid.y >= 0 && grid.y + grid.height <= 768, 'All 63 layers fit in the initial desktop viewport');
    assert((await page.locator('.readout-detail').boundingBox()).width <= 300);
    assert.equal(await page.locator('.gloss').evaluateAll(nodes => nodes.every(n => n.textContent.trim().slice(1, -1).trim().split(/\s+/).length <= 5)), true);
    const answer = () => page.locator('#raw-response').textContent();
    assert.equal(await answer(), 'control-q');
    assert.equal(await page.locator('#alpha-select').isDisabled(), true);
    await page.selectOption('#character-select', 'demon');
    assert.equal(await answer(), 'q-middle-0');
    await page.selectOption('#alpha-select', '2');
    assert.equal(await answer(), 'q-middle-2');
    await page.locator('.token.response').hover();
    assert.match(await page.locator('#selected-position').textContent(), /position 1/);
    assert.equal(await page.locator('.layer-word').first().textContent(), 'cruel');
    // Pointerover updates the real table synchronously, with no tooltip delay.
    await page.locator('.layer[data-layer="1"]').dispatchEvent('pointerover');
    assert.equal(await page.locator('#readout-rows tr').count(), 10);
    assert.match(await page.locator('#readout-caption').textContent(), /Layer 1 · position 1 · top 10/);
    assert.match(await page.locator('#readout-rows tr').first().textContent(), /教练.*coach/);
    assert.deepEqual(await page.locator('#readout-rows tr').first().locator('.numeric').allTextContents(), ['10', '0.1']);
    assert.equal(await page.locator('#readout-rows tr').nth(2).locator('.numeric').last().textContent(), '0.0333333');
    assert.equal(await page.locator('#readout-rows tr').nth(2).locator('.numeric').last().getAttribute('data-value'), String(0.1 / 3));
    assert.match(await page.locator('#readout-rows tr').nth(1).textContent(), /approx\./);
    assert.match(await page.locator('#readout-rows tr').nth(2).textContent(), /untranslated/);
    assert.equal(await page.locator('.layer[title]').count(), 0);
    assert.equal(await page.locator('#readout-rows script').count(), 0);
    await page.fill('#token-search', 'coach');
    assert.equal(await page.locator('.token[data-hit="true"]').count(), 2);
    assert.equal(await page.locator('#readout-rows tr.is-hit').count(), 1);
    await page.fill('#token-search', '教练');
    assert.equal(await page.locator('.token[data-hit="true"]').count(), 2);
    await page.locator('.layer[data-layer="0"]').focus();
    assert.equal(await page.locator('#readout-rows tr').count(), 1);
    assert.match(await page.locator('#readout-caption').textContent(), /Layer 0/);
    await page.fill('#token-search', 'danger');
    assert.equal(await page.locator('.token[data-hit="true"]').count(), 1);
    await page.selectOption('#scheme-select', 'floor');
    assert.match(await answer(), /No generated response/);
    await page.fill('#token-search', 'cruel'); // No exception when a cell failed.
    assert.equal(await page.locator('.token').count(), 0);
    assert.equal(await page.locator('#readout-rows tr').count(), 0);
    await page.selectOption('#question-select', 'other');
    assert.equal(await answer(), 'other-floor-2');
    assert.match(await page.locator('#steering-description').textContent(), /threshold/);
    assert.equal(await page.locator('.token[data-hit="true"]').count(), 1);
    await page.selectOption('#character-select', 'unsteered');
    assert.equal(await answer(), 'control-other');
    assert.equal(await page.locator('#scheme-select').isDisabled(), true);
    await page.selectOption('#character-select', 'demon');
    assert.equal(await answer(), 'other-floor-2');
    await page.screenshot({path: process.argv[2] + '.png', fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    await page.locator('.layer[data-layer="1"]').click();
    assert.equal(await page.locator('#readout-rows tr').count(), 10);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({path: process.argv[2] + '.mobile.png', fullPage: true});
    await page.goto(pathToFileURL(process.argv[3]).href);
    assert.equal(await answer(), 'answer');
    assert.equal(await page.locator('#alpha-select').inputValue(), '3');
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
