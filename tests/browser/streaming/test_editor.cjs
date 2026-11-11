// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const {chromium} = require('playwright-core');
const assert = require('node:assert/strict');
const {startServer} = require('./server.cjs');

async function main() {
    const [channel = 'msedge', backend = 'd3d12', mode = 'headless', application = 'simple'] = process.argv.slice(2);
    const server = await startServer([
        '--stream', ...(mode === 'headless' ? ['--headless'] : []), '--device-type', backend,
        '--width', '960', '--height', '540',
        ...(application === 'usd_stage' ? [] : ['--scene-path', 'data/scenes/emissive-strength.py']),
    ], [`examples/pathtracer/${application}.py`], {portOption: '--stream-port', health: true});
    const watchdog = setTimeout(() => server.child.kill(), 150000);
    let browser, page;
    try {
        const {url} = await server.ready;
        const base = new URL(url).origin;
        browser = await chromium.launch({channel, headless: true});
        page = await browser.newPage({viewport: {width: 1100, height: 750}});
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.goto(url);
        await page.waitForFunction(() => window.falcorStreaming.stats.browser.framesDecoded > 10);
        await page.waitForFunction(() => window.falcorStreaming.stats.server.input_sequence >= 0);
        const stats = () => page.evaluate(() => window.falcorStreaming.stats.server);
        assert.equal((await stats()).mode, 'editor');
        assert.equal((await stats()).editor_visible, true);
        console.log('Editor video decoded');
        // Fit fills the main view.
        for (const windowWidth of [2200, 800, 1100]) {
            await page.setViewportSize({width: windowWidth, height: 750});
            const dimensions = await page.locator('#video').evaluate(video => {
                const rect = video.getBoundingClientRect();
                const viewport = document.getElementById('viewport');
                return [video.videoWidth, video.videoHeight,
                    rect.width === viewport.clientWidth, rect.height === viewport.clientHeight];
            });
            assert.deepEqual(dimensions, [960, 540, true, true]);
        }
        await page.screenshot({path: `build/streaming-deps/editor-${backend}-${mode}.png`});
        const pixels = () => page.evaluate(() => {
            const canvas = document.createElement('canvas');
            canvas.width = 96; canvas.height = 54;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(document.querySelector('video'), 0, 0, 96, 54);
            return Array.from(ctx.getImageData(0, 0, 96, 54).data);
        });
        const docked = await pixels();
        await page.locator('#video').click({position: {x: 200, y: 180}});
        await page.keyboard.press('F5');
        await page.waitForFunction(() => window.falcorStreaming.stats.server.editor_visible === false);
        // Status can arrive before the corresponding video frame is decoded.
        let changed = false;
        for (let i = 0; i < 100; ++i) {
            const fullscreen = await pixels();
            if (docked.some((v, i) => Math.abs(v - fullscreen[i]) > 40)) { changed = true; break; }
            await page.waitForTimeout(50);
        }
        assert(changed, 'UI toggle did not change decoded pixels');
        const initial = (await stats()).camera_position;
        await page.mouse.down({button: 'right'});
        await page.keyboard.down('w');
        await page.waitForFunction(initial => {
            const p = window.falcorStreaming.stats.server.camera_position;
            return p.some((value, i) => Math.abs(value - initial[i]) > 0.2);
        }, initial);
        // Losing focus while controls are held must stop camera movement.
        await page.getByRole('button', {name: 'Disconnect', exact: true}).focus();
        await page.waitForFunction(() => {
            const s = window.falcorStreaming.stats.server;
            return s.pressed_keys.length === 0 && s.pressed_buttons.length === 0;
        });
        await page.waitForTimeout(500);
        const moved = (await stats()).camera_position;
        await page.waitForTimeout(500);
        assert.deepEqual((await stats()).camera_position, moved);
        await page.keyboard.up('w');
        await page.mouse.up({button: 'right'});
        const rendered = (await stats()).rendered;
        await page.getByRole('button', {name: 'Disconnect', exact: true}).click();
        let released = false;
        for (let i = 0; i < 100; ++i) {
            const health = await (await fetch(base + '/health')).json();
            if (health.viewers === 0) { released = true; break; }
            await page.waitForTimeout(50);
        }
        assert(released, 'Disconnect stranded the viewer lease');
        await page.getByRole('button', {name: 'Connect', exact: true}).click();
        await page.waitForFunction(() => window.falcorStreaming.stats.browser.framesDecoded > 10);
        assert.deepEqual((await stats()).camera_position, moved, 'Reconnect reset the application');
        assert.equal((await stats()).editor_visible, false);
        assert((await stats()).rendered > rendered);
        await page.locator('#video').click({position: {x: 200, y: 180}});
        await page.keyboard.press('F12');
        if (backend === 'cuda') {
            await page.waitForTimeout(600);
            assert.equal((await stats()).render_mode, 'path_traced');
        } else {
            await page.waitForFunction(() => window.falcorStreaming.stats.server.render_mode === 'shaded');
        }
        await page.keyboard.press('F5');
        await page.waitForFunction(() => window.falcorStreaming.stats.server.editor_visible === true);
        await page.screenshot({path: `build/streaming-deps/editor-${backend}-${mode}.png`});
        // Escape must leave the application and viewer running.
        const frames = await page.evaluate(() => window.falcorStreaming.stats.browser.framesDecoded);
        await page.keyboard.press('Escape');
        await page.waitForFunction(previous => window.falcorStreaming.stats.browser.framesDecoded > previous + 5, frames);
        assert.equal((await (await fetch(base + '/health')).json()).viewers, 1);
        assert(!server.logs.some(line => line.includes('Session error:')), server.logs.join('\n'));
        assert.deepEqual(errors, []);
        console.log(`Editor streaming passed: ${backend}, ${mode}`);
    } catch (error) {
        if (page) {
            console.error(await page.evaluate(() => window.falcorStreaming.stats));
            await page.screenshot({path: `build/streaming-deps/editor-${backend}-${mode}-failure.png`});
        }
        console.error(server.logs.join('\n'));
        throw error;
    } finally {
        clearTimeout(watchdog);
        if (browser) await browser.close();
        if (server.child.exitCode === null) server.child.kill();
        await server.exited;
    }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
