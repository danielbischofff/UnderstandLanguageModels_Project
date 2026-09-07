// Real browser regression check, invoked by test_report.py when Playwright is available.
const assert = require('node:assert/strict');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright');

(async () => {
  const browser = await chromium.launch(process.env.BROWSER_EXECUTABLE
    ? {executablePath: process.env.BROWSER_EXECUTABLE, headless: true}
    : {channel: 'chrome', headless: true});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(pathToFileURL(process.argv[2]).href);
    const answer = () => page.locator('#raw-response').textContent();
    assert.equal(await answer(), 'control-q');
    assert.equal(await page.locator('#alpha-select').isDisabled(), true);
    await page.selectOption('#character-select', 'demon');
    assert.equal(await answer(), 'q-middle-0');
    await page.selectOption('#alpha-select', '2');
    assert.equal(await answer(), 'q-middle-2');
    await page.locator('.token.response').hover();
    assert.match(await page.locator('#selected-position').textContent(), /position 1/);
    assert.equal(await page.locator('.layer-word').textContent(), 'cruel');
    await page.fill('#token-search', 'danger');
    assert.equal(await page.locator('.token[data-hit="true"]').count(), 1);
    await page.selectOption('#scheme-select', 'floor');
    assert.match(await answer(), /No generated response/);
    await page.fill('#token-search', 'cruel'); // No exception when a cell failed.
    assert.equal(await page.locator('.token').count(), 0);
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
    await page.goto(pathToFileURL(process.argv[3]).href);
    assert.equal(await answer(), 'answer');
    assert.equal(await page.locator('#alpha-select').inputValue(), '3');
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
