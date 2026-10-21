// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const {EventEmitter} = require('node:events');
const fs = require('node:fs');
const path = require('node:path');
const {PassThrough} = require('node:stream');
const {test} = require('node:test');
const vm = require('node:vm');

function harness(outcomes, responses = []) {
    const children = [];
    const spawn = (command, args) => {
        // Every failed child must be fully reaped before a retry starts.
        assert(children.every(child => child.closed));
        const outcome = outcomes[children.length];
        assert(outcome, 'Unexpected startup retry');
        const child = Object.assign(new EventEmitter(), {
            stdout: new PassThrough(), stderr: new PassThrough(),
            exitCode: null, signalCode: null, closed: false,
        });
        child.kill = () => {
            child.signalCode = 'SIGTERM';
            setImmediate(() => {
                child.stdout.end(); child.stderr.end(); child.closed = true;
                child.emit('close', null);
            });
            return true;
        };
        children.push(child);
        setImmediate(() => {
            if (outcome === 'spawn_error') {
                child.emit('error', new Error('Python not found'));
            } else if (outcome === 'health') {
                child.stdout.write('Open http://127.0.0.1:8080/#token=secret to control the editor.\n');
            } else {
                const message = outcome === 'ready' ?
                    {ready: true, url: `http://127.0.0.1:${args[args.indexOf('--port') + 1]}/#token=secret`} :
                    {ready: false, code: outcome, diagnostic: `Failure: ${outcome}`};
                child.stdout.write(JSON.stringify(message) + '\n');
            }
        });
        return child;
    };
    const context = vm.createContext({
        module: {exports: {}}, process, setTimeout, clearTimeout, AbortSignal,
        fetch: async url => {
            assert.match(url, /^http:\/\/127\.0\.0\.1:\d+\/health$/);
            const response = responses.shift();
            assert(response, 'Unexpected health request');
            if (response instanceof Error) throw response;
            return {ok: true, json: async () => response};
        },
        require: name => name === 'node:child_process' ? {spawn} : require(name),
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, 'server.cjs'), 'utf8'), context);
    return {start: context.module.exports.startServer, children};
}

test('port collisions retry with a new server and keep readiness credentials out of logs', async () => {
    const {start, children} = harness(['address_in_use', 'ready']);
    const server = await start();
    try {
        assert.equal(children.length, 2);
        assert.equal(server.child, children[1]);
        assert((await server.ready).url.endsWith('#token=secret'));
        assert.equal(server.logs.length, 0);
    } finally {
        server.child.kill();
        await server.exited;
    }
});

test('port collision retries stop after three attempts and close every child', async () => {
    const {start, children} = harness(['address_in_use', 'address_in_use', 'address_in_use']);
    await assert.rejects(start(), /Failure: address_in_use/);
    assert.equal(children.length, 3);
    assert(children.every(child => child.closed));
});

test('editor readiness polls health without JSON output or logging its private URL', async () => {
    const responses = [new Error('Connection refused'), {ready: false}, {ready: true}];
    const {start} = harness(['health'], responses);
    const server = await start([], ['editor.py'], {portOption: '--stream-port', health: true});
    try {
        assert.equal(responses.length, 0);
        assert.match((await server.ready).url, /^http:\/\/127\.0\.0\.1:\d+\/$/);
        assert.equal(server.logs.length, 0);
    } finally {
        server.child.kill();
        await server.exited;
    }
});

for (const outcome of ['startup_error', 'spawn_error']) {
    test(`${outcome} is cleaned up without retrying`, async () => {
        const {start, children} = harness([outcome]);
        await assert.rejects(start(), /Failure: startup_error|Python not found/);
        assert.equal(children.length, 1);
        assert(children[0].closed);
    });
}
