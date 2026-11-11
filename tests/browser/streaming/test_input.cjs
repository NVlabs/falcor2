// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Exercise DOM capture over a real ordered WebRTC data channel, without a GPU
// or camera consumer. Synthetic composition events cover both commit orderings.
const {chromium} = require('playwright-core');
const assert = require('node:assert/strict');
const path = require('node:path');

async function main() {
    const browser = await chromium.launch({channel: process.argv[2] || 'msedge', headless: true});
    try {
        const page = await browser.newPage({viewport: {width: 1000, height: 700}});
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.setContent('<video id="video" tabindex="0" style="width:640px;height:360px"></video>' +
            '<textarea id="text-input" style="position:absolute;opacity:0;width:1px;height:1px"></textarea>' +
            '<button id="capture">Capture</button><input id="other"><p id="message"></p>');
        await page.addScriptTag({path: path.resolve('falcor2/streaming/web/input.js')});
        await page.evaluate(async () => {
            const a = new RTCPeerConnection(), b = new RTCPeerConnection();
            window.connections = [a, b]; window.received = [];
            a.onicecandidate = e => { if (e.candidate) b.addIceCandidate(e.candidate); };
            b.onicecandidate = e => { if (e.candidate) a.addIceCandidate(e.candidate); };
            b.ondatachannel = e => { e.channel.onmessage = m => received.push(JSON.parse(m.data)); };
            const channel = a.createDataChannel('falcor-input-v1');
            const opened = new Promise(resolve => channel.onopen = resolve);
            await a.setLocalDescription(await a.createOffer());
            await b.setRemoteDescription(a.localDescription);
            await b.setLocalDescription(await b.createAnswer());
            await a.setRemoteDescription(b.localDescription);
            await opened;
            window.remote = new FalcorInput(channel, {generation: 'test', width: 1280, height: 720});
            // The harness owns capture; the production client has no capture button.
            document.getElementById('capture').onclick = () => {
                document.getElementById('text-input').focus();
                document.getElementById('video').requestPointerLock();
            };
        });
        const events = async () => {
            await page.waitForTimeout(80);
            return page.evaluate(() => received.filter(e => e.type !== 'heartbeat'));
        };
        await page.evaluate(() => {
            const video = document.getElementById('video');
            video.dispatchEvent(new MouseEvent('mousemove', {clientX: 100, clientY: 100, bubbles: true}));
            video.focus();
        });
        // The first animation frame/data-channel delivery may exceed 80 ms on
        // a cold browser. Wait for the event rather than sampling an empty list.
        await page.waitForFunction(() => received.some(event => event.type === 'motion'));
        let initial = await events();
        assert.equal(initial[0].type, 'motion');
        for (const key of ['F1', 'F12', 'Home', 'End', 'PageUp', 'PageDown', 'Insert', 'Delete',
            'Tab', 'Backspace', 'Enter', 'Numpad1', 'Semicolon']) await page.keyboard.press(key);
        await page.keyboard.down('a');
        await page.keyboard.down('a');
        await page.keyboard.up('a');
        await page.keyboard.insertText('Hello \u65e5\ud83d\ude42');
        await page.evaluate(() => {
            const text = document.getElementById('text-input');
            text.dispatchEvent(new KeyboardEvent('keydown', {
                code: 'KeyE', key: '\u20ac', ctrlKey: true, altKey: true, modifierAltGraph: true,
                modifierCapsLock: true, modifierNumLock: true, modifierScrollLock: true, bubbles: true,
            }));
        });
        let received = await events();
        for (const code of ['F1', 'F12', 'Home', 'End', 'PageUp', 'PageDown', 'Insert', 'Delete',
            'Tab', 'Backspace', 'Enter', 'Numpad1', 'Semicolon'])
            assert(received.some(e => e.type === 'key' && e.code === code && e.down), code);
        assert(received.some(e => e.type === 'key' && e.code === 'KeyA' && e.repeat));
        assert(received.some(e => e.type === 'key' && e.code === 'Numpad1'));
        assert(received.some(e => e.type === 'key' && e.code === 'KeyE' && e.mods === 6));
        assert(received.some(e => e.type === 'text' && e.text.includes('Hello \u65e5\ud83d\ude42')));

        // Composition input may precede or follow compositionend. Commit once.
        await page.evaluate(() => {
            const text = document.getElementById('text-input');
            for (const final_first of [true, false]) {
                const value = final_first ? '\u65e5' : '\u672c';
                text.dispatchEvent(new CompositionEvent('compositionstart', {data: ''}));
                text.dispatchEvent(new CompositionEvent('compositionupdate', {data: value}));
                const input = () => text.dispatchEvent(new InputEvent('input', {
                    data: value, inputType: 'insertCompositionText', isComposing: final_first,
                }));
                if (final_first) input();
                text.dispatchEvent(new CompositionEvent('compositionend', {data: value}));
                if (!final_first) input();
            }
        });
        received = await events();
        for (const text of ['\u65e5', '\u672c'])
            assert.equal(received.filter(e => e.type === 'text' && e.text === text).length, 1);
        assert.equal(received.filter(e => e.type === 'composition').length, 0);
        await page.keyboard.insertText('\ud83d\ude42'.repeat(600));
        received = await events();
        const chunks = received.filter(e => e.type === 'text' && /^\ud83d\ude42+$/u.test(e.text));
        assert.deepEqual(chunks.map(e => Array.from(e.text).length), [512, 88]);

        // Motion, supported mouse buttons, chords, and wheel units.
        await page.mouse.move(100, 100);
        await page.mouse.down({button: 'left'});
        await page.mouse.down({button: 'right'});
        await page.mouse.up({button: 'left'});
        await page.mouse.up({button: 'right'});
        await page.mouse.click(120, 120, {button: 'middle'});
        const cdp = await page.context().newCDPSession(page);
        for (const [button, buttons] of [['back', 8], ['forward', 16]]) {
            await cdp.send('Input.dispatchMouseEvent', {type: 'mousePressed', x: 120, y: 120, button, buttons, clickCount: 1});
            await cdp.send('Input.dispatchMouseEvent', {type: 'mouseReleased', x: 120, y: 120, button, buttons: 0, clickCount: 1});
        }
        await page.mouse.dblclick(120, 120);
        await page.evaluate(() => {
            const video = document.getElementById('video');
            for (const mode of [0, 1, 2]) video.dispatchEvent(new WheelEvent('wheel', {
                clientX: 120, clientY: 120, deltaX: 1, deltaY: -2, deltaZ: 3,
                deltaMode: mode, metaKey: true, bubbles: true, cancelable: true,
            }));
            // A modifier transition must not be swallowed by motion coalescing.
            video.dispatchEvent(new MouseEvent('mousemove', {clientX: 130, clientY: 120, bubbles: true}));
            video.dispatchEvent(new MouseEvent('mousemove', {clientX: 140, clientY: 120, shiftKey: true, bubbles: true}));
            remote.flush();
        });
        received = await events();
        assert(received.some(e => e.type === 'motion' && !e.relative));
        for (let button = 0; button <= 2; ++button) {
            assert(received.some(e => e.type === 'button' && e.button === button && e.down));
            assert(received.some(e => e.type === 'button' && e.button === button && !e.down));
        }
        assert(!received.some(e => e.type === 'click' || (e.type === 'button' && e.button > 2)));
        for (let mode = 0; mode <= 2; ++mode)
            assert(received.some(e => e.type === 'wheel' && e.mode === mode &&
                e.x === 1 && e.y === -2 && e.mods === 0));
        const motion = received.filter(e => e.type === 'motion').slice(-2);
        assert.deepEqual(motion.map(e => e.mods), [0, 1]);

        // Trusted pointer lock, plus real button transitions while locked.
        await page.locator('#capture').click();
        await page.waitForFunction(() => document.pointerLockElement === document.getElementById('video'));
        const start = (await events()).length;
        await page.mouse.move(200, 180);
        // Recovery must preserve pointer mode even without another capture event.
        await page.evaluate(() => remote.release());
        await page.mouse.down({button: 'left'});
        await page.mouse.up({button: 'left'});
        await page.evaluate(() => document.getElementById('video').dispatchEvent(new WheelEvent('wheel', {
            clientX: 120, clientY: 120, deltaY: 1, bubbles: true, cancelable: true,
        })));
        await page.evaluate(() => document.exitPointerLock());
        received = (await events()).slice(start);
        assert(received.some(e => e.type === 'motion' && e.relative));
        assert(received.some(e => e.type === 'button' && e.button === 0 && e.down && e.relative));
        assert(received.some(e => e.type === 'button' && e.button === 0 && !e.down && e.relative));
        assert(received.some(e => e.type === 'wheel' && e.relative));
        assert(received.some(e => e.type === 'capture' && !e.enabled));

        // A collapsed video must release held controls, never omit coordinates
        // or strand a button until a heartbeat timeout that may never happen.
        await page.locator('#video').hover();
        await page.mouse.down({button: 'left'});
        await page.keyboard.down('w');
        const collapsedStart = (await events()).length;
        await page.evaluate(() => {
            remote.video.style.width = '0px'; remote.video.style.height = '0px';
        });
        await page.mouse.up({button: 'left'});
        const collapsedEvents = (await events()).slice(collapsedStart);
        assert.equal(collapsedEvents.at(-1).type, 'release');
        assert(!collapsedEvents.some(e => e.type === 'button'));
        assert.equal(await page.evaluate(() => remote.buttons.size + remote.pressed.size), 0);
        await page.keyboard.up('w');
        await page.evaluate(() => {
            remote.video.style.width = '640px'; remote.video.style.height = '360px';
        });

        await page.locator('#video').focus();
        await page.keyboard.down('w');
        await page.locator('#other').focus();
        await page.keyboard.up('w');
        const before = (await events()).filter(e => e.type === 'key').length;
        await page.keyboard.type('local');
        received = await events();
        assert.equal(received.filter(e => e.type === 'key').length, before);
        assert(received.some(e => e.type === 'release'));
        assert.equal(await page.evaluate(() => remote.pressed.size), 0);

        // Saturate the sender without flooding the network; the real channel
        // still carries the recovery reset. No lost transition is replayed.
        await page.evaluate(() => {
            Object.defineProperty(remote.channel, 'bufferedAmount', {value: 70000, configurable: true});
            remote.send('motion', {relative: true, x: 2, y: 3, mods: 0});
            delete remote.channel.bufferedAmount;
            remote.send('heartbeat');
        });
        received = await events();
        assert.equal(received.at(-1).type, 'release');
        assert(received.every((e, i) => i === 0 || e.seq > received[i - 1].seq));
        assert.deepEqual(errors, []);
        await page.evaluate(() => { remote.close(); connections.forEach(c => c.close()); });
        console.log(JSON.stringify({success: true, browser: browser.version(), events: received.length}));
    } finally {
        await browser.close();
    }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
