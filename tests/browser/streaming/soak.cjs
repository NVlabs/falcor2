// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Real live playback: ten minutes plus twenty reconnects by default.
// STREAMING_SOAK_URL may contain a remote authenticated URL; never write it to reports.
const {chromium} = require('playwright-core');
const {execFileSync} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

const {startServer} = require('./server.cjs');

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function main() {
    const [channel = 'msedge', output = 'output/streaming-soak', backend = 'automatic', durationArg = '600', reconnectArg = '20'] = process.argv.slice(2);
    const duration = Number(durationArg), reconnects = Number(reconnectArg);
    assert(Number.isFinite(duration) && duration >= 10 && duration <= 86400);
    assert(Number.isInteger(reconnects) && reconnects >= 0 && reconnects <= 1000);
    fs.mkdirSync(output, {recursive: true});
    fs.writeFileSync(path.join(output, 'samples.jsonl'), '');
    const python = process.env.STREAMING_PYTHON || 'python';
    let child, browser, page, watchdog;
    let launchUrl = process.env.STREAMING_SOAK_URL;
    const report = {success: false, channel, backend, requestedSeconds: duration, requestedReconnects: reconnects,
        completedReconnects: 0, samples: [], idleResources: [], limits: {rss_bytes: 256 * 1024 * 1024, threads: 8, os_resources: 32, gpu_memory_mib: 128}};
    const resource = () => child ? JSON.parse(execFileSync(python,
        [path.join(__dirname, '../../python/streaming/process_sample.py'), String(child.pid)], {encoding: 'utf8', windowsHide: true, timeout: 10000})) : null;
    const checkGrowth = (baseline, current) => {
        if (!baseline || !current) return;
        for (const [name, limit] of Object.entries(report.limits)) {
            if (baseline[name] !== null && current[name] !== null)
                assert(current[name] - baseline[name] <= limit, `${name} growth exceeded ${limit}: ${baseline[name]} -> ${current[name]}`);
        }
    };
    try {
        if (!launchUrl) {
            const server = await startServer(['--backend', backend]);
            child = server.child;
            launchUrl = (await server.ready).url;
        }
        report.resourceCoverage = child ? 'local server process; GPU memory only where nvidia-smi supports accounting' : 'remote server resources not sampled';
        watchdog = setTimeout(() => { if (child) child.kill(); if (browser) browser.close(); }, (duration + reconnects * 30 + 180) * 1000);
        const base = new URL(launchUrl).origin;
        for (let attempt = 0; attempt < 100; ++attempt) {
            try { if ((await fetch(base + '/health', {signal: AbortSignal.timeout(5000)})).ok) break; } catch (_) { }
            await sleep(100);
        }
        browser = await chromium.launch({headless: true, ...(channel === 'chromium' ? {} : {channel})});
        report.browserVersion = browser.version();
        page = await browser.newPage();
        const pageErrors = [];
        page.on('pageerror', error => pageErrors.push(error.message));
        await page.goto(launchUrl);
        const snapshot = () => page.evaluate(() => structuredClone(window.falcorStreaming.stats));
        const connect = async (automatic = false) => {
            if (!automatic) await page.getByRole('button', {name: 'Connect', exact: true}).click();
            await page.waitForFunction(() => window.falcorStreaming.stats.browser.framesDecoded >= 30 &&
                window.falcorStreaming.stats.server.encoded_bytes > 0, null, {timeout: 30000});
            const stats = await snapshot();
            assert.equal(stats.server.encoder, 'openh264');
            assert.equal(stats.server.mode, 'pattern', 'Soak throughput acceptance requires --demo pattern');
            assert.equal(stats.browser.codec.toLowerCase(), 'video/h264');
            assert.equal(stats.browser.width, 1280);
            assert.equal(stats.browser.height, 720);
        };
        const disconnect = async () => {
            await page.getByRole('button', {name: 'Disconnect', exact: true}).click();
            for (let i = 0; i < 200; ++i) {
                if (!(await (await fetch(base + '/health', {signal: AbortSignal.timeout(5000)})).json()).busy) return;
                await sleep(50);
            }
            throw new Error('Disconnect stranded server lease');
        };
        // Warm driver/compiler/codec caches and transport threads before baselines.
        for (let i = 0; i < 3; ++i) { await connect(i === 0); await disconnect(); }
        await sleep(1000);
        const idleBaseline = resource();
        report.idleResources.push(idleBaseline);
        await connect();
        await sleep(5000);
        const activeBaseline = resource();
        report.activeBaseline = activeBaseline;
        report.gpuMemoryAvailable = activeBaseline?.gpu_memory_mib != null;
        const first = await snapshot();
        await page.screenshot({path: path.join(output, 'diagnostics.png'), fullPage: true});
        const started = Date.now();
        let previous = first;
        while ((Date.now() - started) / 1000 < duration) {
            await sleep(Math.min(10000, duration * 1000 - (Date.now() - started)));
            const current = await snapshot();
            assert.equal(current.server.failed, false);
            assert.equal(current.server.connected, true);
            assert(current.browser.framesDecoded > previous.browser.framesDecoded, 'Decoded frames stopped advancing');
            assert(current.server.elapsed_seconds > previous.server.elapsed_seconds, 'Server diagnostics stopped updating');
            assert(current.server.encoded_bytes > previous.server.encoded_bytes, 'Encoder stopped advancing');
            assert(current.server.queue_bytes <= 2 * 1024 * 1024 && current.server.queue_frames <= 8);
            assert(current.server.encoded_bitrate > 0 && current.server.encode_wall_ms > 0);
            for (const field of ['gpu_prepare_ms', 'encode_wall_ms', 'submit_to_output_ms', 'encoded_bitrate'])
                assert(Number.isFinite(current.server[field]), `Invalid diagnostic ${field}`);
            assert(current.server.submit_to_output_ms >= current.server.encode_wall_ms);
            const resources = resource();
            const sample = {seconds: (Date.now() - started) / 1000, ...current, resources};
            report.samples.push(sample);
            fs.appendFileSync(path.join(output, 'samples.jsonl'), JSON.stringify(sample) + '\n');
            checkGrowth(activeBaseline, resources);
            if (report.samples.length % 6 === 0) console.log(JSON.stringify({progressSeconds: sample.seconds, encodedMbps: current.server.encoded_bitrate / 1e6}));
            previous = current;
        }
        report.elapsedSeconds = (Date.now() - started) / 1000;
        const seconds = previous.server.elapsed_seconds - first.server.elapsed_seconds;
        report.encodedFps = (previous.server.encoded - first.server.encoded) / seconds;
        report.decodedFps = (previous.browser.framesDecoded - first.browser.framesDecoded) / report.elapsedSeconds;
        report.encodedMbps = (previous.server.encoded_bytes - first.server.encoded_bytes) * 8 / seconds / 1e6;
        report.resourcePeaks = Object.fromEntries(Object.keys(report.limits).map(name => {
            const values = report.samples.map(sample => sample.resources?.[name]).filter(value => value != null);
            return [name, values.length ? Math.max(...values) : null];
        }));
        assert(report.encodedFps >= 28 && report.decodedFps >= 27, '720p30 pattern throughput target missed');
        await disconnect();
        for (let i = 0; i < reconnects; ++i) {
            await connect(); await disconnect(); await sleep(250);
            const resources = resource();
            report.idleResources.push(resources);
            report.completedReconnects += 1;
            checkGrowth(idleBaseline, resources);
        }
        assert.deepEqual(pageErrors, []);
        report.success = true;
        console.log(JSON.stringify({success: true, elapsedSeconds: report.elapsedSeconds,
            encodedFps: report.encodedFps, decodedFps: report.decodedFps, encodedMbps: report.encodedMbps, reconnects: report.completedReconnects}));
    } catch (error) {
        // Avoid leaking authenticated URLs through Playwright error messages.
        report.failure = String(error.message).split(launchUrl || '__no_url__').join('[redacted URL]');
        throw new Error(report.failure);
    } finally {
        clearTimeout(watchdog);
        if (browser) await browser.close();
        // Connections were drained and sampled first. Process termination is
        // harness cleanup, not evidence of graceful application shutdown.
        if (child && child.exitCode === null) child.kill();
        fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2) + '\n');
    }
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
