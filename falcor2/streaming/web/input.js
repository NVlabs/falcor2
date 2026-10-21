// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Capture browser input without camera mappings. Only adjacent motion with the
// same modifiers/buttons is coalesced; relative displacement is accumulated.
window.FalcorInput = class {
    constructor(channel, config) {
        this.channel = channel; this.config = config; this.seq = 0;
        this.pending = null; this.pressed = new Set(); this.buttons = new Set();
        this.blocked = false; this.listeners = []; this.composing = false;
        this.committed_text = null;
        const video = this.video = document.getElementById('video');
        const text = this.text = document.getElementById('text-input');
        const on = (target, name, handler, options) => {
            target.addEventListener(name, handler, options);
            this.listeners.push(() => target.removeEventListener(name, handler, options));
        };
        const mods = event => (event.shiftKey ? 1 : 0) | (event.ctrlKey ? 2 : 0) | (event.altKey ? 4 : 0);
        const focused = () => document.activeElement === text || document.activeElement === video;
        const locked = () => document.pointerLockElement === video;
        const focus = () => text.focus({preventScroll: true});
        on(video, 'focus', focus);
        on(text, 'blur', () => {
            this.release();
            if (locked()) document.exitPointerLock();
        });
        on(window, 'blur', () => { this.release(); });
        on(document, 'visibilitychange', () => {
            if (document.hidden) { this.release(); }
        });

        // Mouse events report every button transition, including chords and
        // transitions while pointer-locked (pointerdown/up alone do not).
        on(video, 'mousedown', event => {
            const pos = this.position(event, locked());
            if (!pos || event.button > 2) return;
            event.preventDefault(); focus();
            if (this.send('button', {button: event.button, down: true, relative: locked(), ...pos, mods: mods(event)}))
                this.buttons.add(event.button);
        });
        on(document, 'mouseup', event => {
            if ((!this.buttons.has(event.button) && !locked()) || event.button > 2) return;
            event.preventDefault();
            const pos = this.position(event, true);
            if (!pos) { this.release(); return; }
            this.send('button', {button: event.button, down: false, relative: locked(), ...pos, mods: mods(event)});
            this.buttons.delete(event.button);
        });
        on(document, 'mousemove', event => {
            const relative = locked();
            if (!relative && event.target !== video && this.buttons.size === 0) return;
            let pos;
            if (relative) {
                const rect = this.rectangle();
                if (!rect.width || !rect.height) return;
                pos = {x: event.movementX * config.width / rect.width, y: event.movementY * config.height / rect.height};
            } else pos = this.position(event, this.buttons.size !== 0);
            if (!pos) return;
            const modifiers = mods(event);
            if (this.pending && (this.pending.relative !== relative ||
                this.pending.buttons !== event.buttons || this.pending.mods !== modifiers)) this.flush();
            if (relative && this.pending) { pos.x += this.pending.x; pos.y += this.pending.y; }
            this.pending = {relative, ...pos, buttons: event.buttons, mods: modifiers};
        });
        on(video, 'pointercancel', () => this.release());
        on(video, 'contextmenu', event => event.preventDefault());
        on(video, 'wheel', event => {
            const pos = this.position(event, locked());
            if (!pos) return;
            event.preventDefault();
            this.send('wheel', {relative: locked(), x: event.deltaX, y: event.deltaY, mode: event.deltaMode,
                px: pos.x, py: pos.y, mods: mods(event)});
        }, {passive: false});

        const key = (event, down) => {
            const identity = event.code || 'Unidentified:' + event.location;
            if (!focused() && !this.pressed.has(identity)) return;
            if (down) this.committed_text = null;
            if (this.send('key', {code: event.code, down, repeat: down && event.repeat, mods: mods(event)})) {
                if (down) this.pressed.add(identity); else this.pressed.delete(identity);
            }
            // Allow the text input to generate text/IME events. Prevent local
            // navigation and shortcuts while the remote surface owns focus.
            if (!down || (!event.isComposing && event.key !== 'Process' && event.key !== 'Dead' &&
                ((event.ctrlKey && !event.getModifierState('AltGraph')) || event.metaKey ||
                    (Array.from(event.key).length !== 1 && event.key !== 'Enter'))))
                event.preventDefault();
        };
        on(window, 'keydown', event => key(event, true));
        on(window, 'keyup', event => key(event, false));
        for (const phase of ['start', 'update', 'end']) on(text, 'composition' + phase, event => {
            if (!focused()) return;
            this.flush();
            this.composing = phase !== 'end';
            if (phase === 'end') {
                this.send_text(event.data || '');
                this.committed_text = event.data || '';
                text.value = '';
                clearTimeout(this.commit_timer);
                this.commit_timer = setTimeout(() => { this.committed_text = null; }, 0);
            }
        });
        on(text, 'input', event => {
            if (!focused() || this.composing || event.isComposing) return;
            const value = event.data ?? (event.inputType === 'insertLineBreak' ? '\n' : text.value);
            // Engines may send the final input before or after compositionend.
            if (value !== this.committed_text) this.send_text(value);
            this.committed_text = null;
            text.value = '';
        });
        on(document, 'pointerlockchange', () => {
            this.release();
            const enabled = locked();
            this.send('capture', {enabled});
        });
        on(channel, 'close', () => this.release());
        this.timer = setInterval(() => { if (!document.hidden) this.send('heartbeat'); }, 250);
        const tick = () => { this.flush(); this.animation = requestAnimationFrame(tick); };
        this.animation = requestAnimationFrame(tick);
    }
    rectangle() {
        const rect = this.video.getBoundingClientRect();
        const scale = Math.min(rect.width / this.config.width, rect.height / this.config.height);
        const width = this.config.width * scale, height = this.config.height * scale;
        return {left: rect.left + (rect.width - width) / 2, top: rect.top + (rect.height - height) / 2, width, height};
    }
    position(event, clamp = false) {
        const rect = this.rectangle();
        if (!rect.width || !rect.height) return null;
        let x = (event.clientX - rect.left) * this.config.width / rect.width;
        let y = (event.clientY - rect.top) * this.config.height / rect.height;
        if (!clamp && (x < 0 || y < 0 || x > this.config.width || y > this.config.height)) return null;
        return {x: Math.max(0, Math.min(this.config.width, x)), y: Math.max(0, Math.min(this.config.height, y))};
    }
    clear_state() {
        this.pending = null; this.pressed.clear(); this.buttons.clear();
        this.composing = false; this.committed_text = null; this.text.value = '';
    }
    send(type, fields = {}) {
        if (type !== 'motion' && type !== 'heartbeat') this.flush();
        if (this.channel.readyState !== 'open') return false;
        if (this.channel.bufferedAmount > 65536) {
            this.blocked = true; this.clear_state();
            if (document.pointerLockElement === this.video) document.exitPointerLock();
            return false;
        }
        const message = (kind, values) => JSON.stringify({v: 1, generation: this.config.generation, seq: ++this.seq, type: kind, ...values});
        try {
            if (this.blocked) {
                this.channel.send(message('release', {})); this.blocked = false;
                // Drop the triggering event: its key/button transition may
                // depend on events lost during overflow.
                return false;
            }
            const payload = message(type, fields);
            if (new TextEncoder().encode(payload).length > 4096) {
                this.clear_state(); this.channel.send(message('release', {})); return false;
            }
            this.channel.send(payload);
            return true;
        } catch (_) { this.blocked = true; this.clear_state(); return false; }
    }
    send_text(value) {
        const characters = Array.from(value);
        for (let i = 0; i < characters.length; i += 512)
            if (!this.send('text', {text: characters.slice(i, i + 512).join('')})) break;
    }
    flush() {
        if (!this.pending) return;
        const value = this.pending; this.pending = null;
        const count = value.relative ? Math.max(1, Math.ceil(Math.max(Math.abs(value.x), Math.abs(value.y)) / 8192)) : 1;
        for (let i = 0; i < Math.min(count, 16); ++i)
            if (!this.send('motion', {relative: value.relative, mods: value.mods, x: value.x / count, y: value.y / count})) break;
        if (count > 16) this.release();
    }
    release() {
        this.flush(); this.clear_state(); this.send('release');
    }
    close() {
        this.release(); clearInterval(this.timer); cancelAnimationFrame(this.animation);
        clearTimeout(this.commit_timer);
        for (const remove of this.listeners) remove();
        if (document.pointerLockElement === this.video) document.exitPointerLock();
    }
};
