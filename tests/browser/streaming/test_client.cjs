// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

function client(decodingInfo, hash = '') {
    const elements = new Map();
    const sockets = [];
    const intervals = [];
    const location = {hash, pathname: '/', protocol: 'http:', host: 'localhost:8080'};
    const context = vm.createContext({
        document: {body: {style: {}}, listeners: {},
            addEventListener(type, callback) { this.listeners[type] = callback; },
            getElementById: id => {
            if (!elements.has(id)) elements.set(id, {style: {}, listeners: {},
                addEventListener(type, callback) { this.listeners[type] = callback; },
                setAttribute(name, value) { this[name] = value; },
                click() { this.listeners.click(); }});
            return elements.get(id);
        }},
        location, history: {replaceState() { location.hash = ''; }},
        window: {addEventListener() {}}, setInterval(callback) { intervals.push(callback); }, URLSearchParams,
        navigator: {mediaCapabilities: decodingInfo ? {decodingInfo} : undefined},
        RTCRtpReceiver: {getCapabilities: () => ({codecs: [{mimeType: 'video/H264'}]})},
        WebSocket: class {
            constructor() { sockets.push(this); this.messages = []; }
            send(value) { this.messages.push(JSON.parse(value)); }
            close() { this.closed = true; }
        },
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../../../falcor2/streaming/web/client.js'), 'utf8'), context);
    return Object.assign(context.create_answer, {context, sockets, elements, intervals});
}

for (const token of ['', 'a&b +/#']) test(`connect and reconnect with ${token ? 'URL token' : 'no token'}`, () => {
    const {context, sockets, elements} = client(undefined, token ? '#' + new URLSearchParams({token}) : '');
    assert.equal(context.location.hash, '');
    assert.equal(sockets.length, 1, 'Opening the viewer must connect once');
    const button = elements.get('connection');
    assert.equal(elements.get('connection-label').textContent, 'Disconnect');
    // The same button can cancel connection setup before the socket opens.
    button.click();
    assert.equal(sockets[0].closed, true);
    assert.equal(elements.get('connection-label').textContent, 'Connect');
    button.click();
    assert.equal(elements.get('connection-label').textContent, 'Disconnect');
    sockets[1].onopen();
    assert.deepEqual(sockets[1].messages, [{type: 'auth', token}]);
    sockets[0].onclose(); // An old connection must not disconnect its replacement.
    assert.equal(elements.get('connection-label').textContent, 'Disconnect');
    assert.equal(sockets[1].closed, undefined);
    button.click();
    assert.equal(sockets[1].closed, true);
    assert.equal(elements.get('connection-label').textContent, 'Connect');
});

const sdp = 'a=rtpmap:102 H264/90000\r\n' +
    'a=fmtp:102 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f\r\n' +
    'a=fmtp:103 apt=102\r\n';
const connection = level => ({createAnswer: async () => ({type: 'answer', sdp}),
    remoteDescription: {sdp: sdp.replace('42e01f', '42c0' + level)}});
const config = {width: 1280, height: 720, bitrate: 8000000, fps: 60};

test('statistics refresh only while open and reset on disconnect', async () => {
    const {context, sockets, elements} = client();
    const panel = elements.get('statistics');
    const values = context.document.getElementById('statistics-values');
    const update = bitrate => sockets[0].onmessage({data: JSON.stringify({type: 'stats', stats: {
        target_bitrate: 8000000, encoded_bitrate: bitrate, capture_skips: 0,
    }})});
    await update(4000000);
    assert.equal(values.textContent, undefined);
    panel.open = true;
    context.update_statistics();
    assert.match(values.textContent, /4.0 \/ 8.0 Mbit\/s/);
    assert.match(values.textContent, /Capture skips: 0/);
    panel.open = false;
    await update(6000000);
    assert.match(values.textContent, /4.0 \/ 8.0 Mbit\/s/);
    panel.open = true;
    context.update_statistics();
    assert.match(values.textContent, /6.0 \/ 8.0 Mbit\/s/);
    context.disconnect();
    assert.match(values.textContent, /-- \/ -- Mbit\/s/);
});

for (const reason of ['busy', 'unauthorized', 'transport', 'closed', 'manual'])
test(`${reason} disconnect waits for a manual retry`, async () => {
    const {context, sockets, elements, intervals} = client();
    const socket = sockets[0];
    if (reason === 'manual') elements.get('connection').click();
    else if (reason === 'transport') socket.onerror();
    else if (reason === 'closed') socket.onclose();
    else await socket.onmessage({data: JSON.stringify({type: 'error', message: reason})});
    socket.onclose();
    const message = elements.get('message').textContent;
    for (let tick = 0; tick < 5; ++tick) for (const interval of intervals) await interval();
    assert.equal(sockets.length, 1, 'Must not automatically retry');
    assert.equal(elements.get('connection-label').textContent, 'Connect');
    assert.equal(elements.get('message').textContent, message);
    if (reason === 'busy' || reason === 'unauthorized') assert.equal(message, reason);
    elements.get('connection').click();
    assert.equal(sockets.length, 2);
    assert.equal(elements.get('connection-label').textContent, 'Disconnect');
    assert.equal(elements.get('message').textContent, '');
});

test('bitrate-driven level 4.1 checks the full requested bitrate', async () => {
    let queried;
    const create_answer = client(async value => { queried = value; return {supported: true}; });
    const answer = await create_answer(connection('29'), {...config, bitrate: 50000000});
    assert.equal(queried.video.bitrate, 50000000);
    assert(queried.video.contentType.includes('profile-level-id=42e029'));
    assert.equal(answer.sdp, sdp.replace('42e01f', '42e029'));
});

test('30 FPS preserves the browser answer without a capability probe', async () => {
    const create_answer = client(() => { throw new Error('Unexpected capability probe'); });
    assert.equal((await create_answer(connection('1f'), {...config, fps: 30})).sdp, sdp);
});

for (const [width, height, fps, level] of [[1280, 720, 60, '20'], [1920, 1080, 30, '28'],
    [1920, 1080, 60, '2a'], [3840, 2160, 30, '33'], [3840, 2160, 60, '34']])
test(`${width}x${height} at ${fps} FPS raises only the confirmed H.264 receive level`, async () => {
    let queried;
    const create_answer = client(async value => { queried = value; return {supported: true}; });
    const answer = await create_answer(connection(level), {...config, width, height, fps});
    assert.equal(queried.type, 'webrtc');
    assert.equal(queried.video.framerate, fps);
    assert.equal(queried.video.width, width);
    assert.equal(queried.video.height, height);
    assert.equal(queried.video.bitrate, 8000000);
    assert(queried.video.contentType.includes('profile-level-id=42e0' + level));
    assert.equal(answer.sdp, sdp.replace('42e01f', '42e0' + level));
});

test('60 FPS rejects unsupported or unavailable capability checks', async () => {
    for (const probe of [undefined, async () => ({supported: false})]) {
        await assert.rejects(client(probe)(connection('20'), config), /1280x720 at 30 FPS and 8 Mbit\/s/);
    }
});
