// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const {spawn} = require('node:child_process');
const {createInterface} = require('node:readline');
const net = require('node:net');

// Launch the production server on an ephemeral port. Keep credential-bearing
// readiness output out of captured diagnostics and reports.
async function startServer(args = [], entry = ['-m', 'falcor2.streaming.server', '--log-format', 'json'], options = {}) {
    for (let attempt = 0; ; ++attempt) {
        const server = await launchServer(args, entry, options);
        try {
            await server.ready;
            return server;
        } catch (error) {
            if (server.child.exitCode === null && server.child.signalCode === null) server.child.kill();
            await server.exited;
            if (error.code !== 'address_in_use' || attempt >= 2) throw error;
        }
    }
}

async function launchServer(args, entry, {portOption = '--port', health = false}) {
    const listener = net.createServer();
    await new Promise((resolve, reject) => {
        listener.once('error', reject);
        listener.listen(0, '127.0.0.1', resolve);
    });
    const port = listener.address().port;
    await new Promise(resolve => listener.close(resolve));
    const child = spawn(process.env.STREAMING_PYTHON || 'python',
        ['-u', '-X', 'faulthandler', ...entry, portOption, String(port), ...args], {windowsHide: true});
    const logs = [];
    let readyResolve, readyReject;
    const ready = new Promise((resolve, reject) => { readyResolve = resolve; readyReject = reject; });
    let pollTimer, settled = false;
    const finish = (error, value) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        clearTimeout(pollTimer);
        if (error) readyReject(error); else readyResolve(value);
    };
    const timeout = setTimeout(() => {
        child.kill();
        finish(new Error('Server startup timed out'));
    }, 60000);
    const fail = error => finish(error);
    createInterface({input: child.stdout}).on('line', line => {
        let value;
        try { value = JSON.parse(line); } catch (_) { }
        if (value?.ready && value.url) { finish(null, value); return; }
        if (value?.ready === false) {
            fail(Object.assign(new Error(value.diagnostic), {code: value.code})); return;
        }
        // Editor examples print a human-readable URL; it may contain a token.
        if (!line.startsWith('Open http')) logs.push(line);
    });
    child.stderr.on('data', data => logs.push(data.toString()));
    child.on('error', fail);
    const exited = new Promise(resolve => child.on('close', code => {
        fail(new Error(`Server exited: ${code}`));
        resolve(code);
    }));
    if (health) {
        const url = `http://127.0.0.1:${port}/`;
        const poll = async () => {
            if (settled) return;
            try {
                const response = await fetch(url + 'health', {signal: AbortSignal.timeout(1000)});
                if (response.ok && (await response.json()).ready) {
                    finish(null, {ready: true, url});
                    return;
                }
            } catch (_) { /* The editor may still be loading its scene. */ }
            if (!settled) pollTimer = setTimeout(poll, 100);
        };
        void poll();
    }
    return {child, ready, exited, logs};
}

module.exports = {startServer};
