// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Runs the actual GPU server and browser UI. No fake encoder or prerecorded video.
const { chromium, firefox } = require('playwright-core');
const assert = require('node:assert/strict');

const {startServer} = require('./server.cjs');

async function disconnect(page, base) {
    await page.getByRole('button', {name: 'Disconnect', exact: true}).click();
    assert.equal(await page.locator('nav button').count(), 1);
    assert.equal(await page.getByRole('button', {name: 'Connect', exact: true}).isVisible(), true);
    for (let i = 0; i < 100; ++i) {
        const health = await (await fetch(base + '/health', {signal: AbortSignal.timeout(5000)})).json();
        if (!health.busy) {
            assert.deepEqual(health.last_session.pressed_keys, []);
            assert.deepEqual(health.last_session.pressed_buttons, []);
            return;
        }
        await page.waitForTimeout(50);
    }
    throw new Error('Disconnect stranded the lease');
}

async function checkCamera(page, base) {
    const stats = () => page.evaluate(() => window.falcorStreaming.stats.server);
    await page.waitForFunction(() => window.falcorStreaming.stats.browser.framesDecoded >= 15);
    const pixels = () => page.evaluate(() => {
        const video = document.querySelector('video');
        const canvas = document.createElement('canvas'); canvas.width = 32; canvas.height = 18;
        const context = canvas.getContext('2d'); context.drawImage(video, 0, 0, 32, 18);
        return Array.from(context.getImageData(0, 0, 32, 18).data);
    });
    const first = await pixels();
    const initial = (await stats()).camera_position;
    await page.locator('#video').hover();
    await page.mouse.down({button: 'right'});
    await page.keyboard.down('w');
    await page.waitForFunction(start => window.falcorStreaming.stats.server.camera_position[2] < start - 0.2, initial[2]);
    await page.mouse.move(450, 280);
    await page.waitForTimeout(200);
    const moved = await pixels();
    if (first.every((v, i) => Math.abs(v - moved[i]) < 10)) throw new Error('Camera input did not change decoded pixels');
    // Losing viewport focus while controls are held must release the controller.
    await page.getByRole('button', {name: 'Disconnect', exact: true}).focus();
    await page.waitForFunction(() => {
        const s = window.falcorStreaming.stats.server;
        return s.pressed_keys.length === 0 && s.pressed_buttons.length === 0;
    });
    const stopped = (await stats()).camera_position;
    await page.waitForTimeout(700);
    if (JSON.stringify(stopped) !== JSON.stringify((await stats()).camera_position)) throw new Error('Camera kept moving after focus loss');
    await page.keyboard.up('w'); await page.mouse.up({button: 'right'});
    await disconnect(page, base);
    await page.getByRole('button', {name: 'Connect', exact: true}).click();
    await page.waitForFunction(() => window.falcorStreaming.stats.browser.framesDecoded >= 15);
    if (JSON.stringify((await stats()).camera_position) !== JSON.stringify(initial)) throw new Error('Reconnect reused camera state');
}

async function main() {
    const [channel = 'msedge', backend = 'automatic', demo = 'pattern'] = process.argv.slice(2);
    const {child, ready, exited, logs} = await startServer(['--run-seconds', '40', '--backend', backend, '--demo', demo,
        ...(process.env.STREAMING_TEST_TOKEN ? ['--token', process.env.STREAMING_TEST_TOKEN] : [])]);
    const watchdog = setTimeout(() => child.kill(), 150000);
    let browser, page;
    try {
        const launch = await ready;
        const base = new URL(launch.url).origin;
        browser = channel === 'firefox' ? await firefox.launch({headless: true}) :
            await chromium.launch({channel, headless: true});
        page = await browser.newPage();
        const pageErrors = [];
        page.on('pageerror', error => pageErrors.push(error.message));
        if (process.env.STREAMING_TEST_TOKEN) {
            const unauthorized = await browser.newPage();
            for (const suffix of ['/', '/#token=wrong']) {
                await unauthorized.goto(base + suffix);
                await unauthorized.waitForFunction(() => document.querySelector('#message').textContent.includes('Invalid session token'));
            }
            await unauthorized.close();
        }
        await page.goto(launch.url);
        if (await page.locator('#capture').count()) throw new Error('Unexpected capture control');
        if (await page.locator('#statistics').evaluate(element => element.open)) throw new Error('Statistics must start closed');
        await page.waitForFunction(() => window.falcorStreaming.stats.browser.framesDecoded >= 60);
        assert.equal(await page.locator('#video').evaluate(video => video.paused), false, 'Video did not autoplay');
        assert.equal(await page.locator('nav button').count(), 1);
        assert.equal(await page.getByRole('button', {name: 'Disconnect', exact: true}).isVisible(), true);
        assert.equal(await page.locator('#statistics-values').textContent(), '');
        await page.locator('#sidebar-toggle').click();
        await page.locator('#statistics summary').click();
        await page.waitForFunction(() => /FPS: [0-9]/.test(document.querySelector('#statistics-values').textContent));
        await page.locator('#statistics summary').click();
        const text = await page.locator('#statistics-values').textContent();
        await page.waitForTimeout(600);
        assert.equal(await page.locator('#statistics-values').textContent(), text, 'Closed statistics kept updating');
        const secondViewer = await browser.newPage();
        await secondViewer.goto(launch.url);
        await secondViewer.waitForFunction(() => document.querySelector('#message').textContent.includes('Another viewer'));
        await secondViewer.close();
        if (demo === 'camera') {
            await checkCamera(page, base);
        } else {
            // Pattern playback negotiates input independently of a camera.
            await page.locator('#video').focus();
            await page.keyboard.down('F12');
            await page.waitForFunction(() => window.falcorStreaming.stats.server.pressed_keys.includes('F12'));
            await page.keyboard.up('F12');
            await page.keyboard.insertText('Browser input');
            const input_sequence = await page.evaluate(() => remoteInput.seq);
            await page.waitForFunction(sequence => window.falcorStreaming.stats.server.input_sequence >= sequence,
                input_sequence);
            const input_stats = await page.evaluate(() => window.falcorStreaming.stats.server);
            if (input_stats.input_rejected || input_stats.pressed_keys.length)
                throw new Error('Pattern input failed validation or left a held key');
            const snapshot = async () => page.evaluate(() => {
                const video = document.querySelector('video');
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth; canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d'); ctx.drawImage(video, 0, 0);
                const pixel = (x, y) => Array.from(ctx.getImageData(x, y, 1, 1).data).slice(0, 3);
                const bars = Array.from({length: 8}, (_, i) => pixel(Math.floor((i + 0.5) * canvas.width / 8), Math.floor(canvas.height / 4)));
                const stripe = [];
                for (let x = 0; x < canvas.width; x += 8) {
                    const rgb = pixel(x, Math.floor(canvas.height * 3 / 4));
                    if (rgb[0] > 220 && rgb[1] < 30 && rgb[2] > 220) stripe.push(x);
                }
                return {stats: window.falcorStreaming.stats, width: canvas.width, height: canvas.height, bars, stripe};
            });
            const first = await snapshot();
            await page.waitForTimeout(300);
            const second = await snapshot();
            if (!first.stripe.length || !second.stripe.length || JSON.stringify(first.stripe) === JSON.stringify(second.stripe))
                throw new Error('GPU pattern did not animate');
            const expected = [[255,255,255],[255,255,0],[0,255,255],[0,255,0],[255,0,255],[255,0,0],[0,0,255],[0,0,0]];
            const colorError = Math.max(...first.bars.flatMap((rgb, i) => rgb.map((v, c) => Math.abs(v - expected[i][c]))));
            if (colorError > 12 || first.width !== 1280 || first.height !== 720) throw new Error('Video color/dimension mismatch');
            if (first.stats.browser.codec.toLowerCase() !== 'video/h264' || first.stats.browser.keyFramesDecoded < 1)
                throw new Error('Missing H.264 keyframe decode');
            if (!first.stats.browser.localCandidateType || !first.stats.browser.remoteCandidateType)
                throw new Error('Selected ICE pair was not reported');
            if (first.stats.server.queue_bytes > 2 * 1024 * 1024 || first.stats.server.queue_frames > 8)
                throw new Error('Native media queue exceeded bounds');
            await disconnect(page, base);
            await page.getByRole('button', {name: 'Connect', exact: true}).click();
        }
        await page.waitForFunction(() => window.falcorStreaming.stats.browser.framesDecoded >= 30,
            null, {timeout: 8000});
        const code = await exited;
        if (code !== 0) throw new Error(`Server shutdown exit code ${code}`);
        assert.deepEqual(pageErrors, []);
        console.log(`Passed ${demo} playback, input, reconnect, and shutdown (${channel}).`);
    } catch (error) {
        const diagnostic = page ? await page.evaluate(() => ({
            message: document.querySelector('#message')?.textContent,
            status: document.querySelector('#status')?.textContent,
            videoWidth: document.querySelector('video')?.videoWidth,
            stats: window.falcorStreaming?.stats,
            codecs: RTCRtpReceiver.getCapabilities('video')?.codecs,
        })).catch(() => ({})) : {};
        throw new Error(`${error.stack}\nBrowser diagnostics: ${JSON.stringify(diagnostic)}\nServer diagnostics:\n${logs.slice(-30).join('\n')}`);
    } finally {
        clearTimeout(watchdog);
        if (browser) await browser.close();
        if (child.exitCode === null) child.kill();
    }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
