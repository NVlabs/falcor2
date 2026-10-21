// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const element = id => document.getElementById(id);
let peer = null, socket = null;
let remoteInput = null;
let browserStats = {}, serverStats = {};
const token = new URLSearchParams(location.hash.slice(1)).get('token') || '';
history.replaceState(null, '', location.pathname);

function update_statistics() {
    if (!element('statistics').open) return;
    const number = (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : '--';
    element('statistics-values').textContent = [
        `FPS: ${number(browserStats.framesPerSecond)}`,
        `Bitrate: ${number(serverStats.encoded_bitrate / 1e6)} / ${number(serverStats.target_bitrate / 1e6)} Mbit/s (actual / target)`,
        `GPU prepare: ${number(serverStats.gpu_prepare_ms, 2)} ms`,
        `Encode: ${number(serverStats.encode_wall_ms, 2)} ms`,
        `Submit to output: ${number(serverStats.submit_to_output_ms, 2)} ms`,
        `Round trip: ${number(browserStats.rttMs)} ms`,
        `Capture skips: ${number(serverStats.capture_skips, 0)}`,
    ].join('\n');
}

function disconnect(message = '') {
    if (remoteInput) { remoteInput.close(); remoteInput = null; }
    if (socket) { socket.close(); socket = null; }
    if (peer) { peer.close(); peer = null; }
    element('video').srcObject = null;
    connection_label('Connect');
    element('message').textContent = message;
    browserStats = {}; serverStats = {};
    update_statistics();
}

async function gather(connection) {
    if (connection.iceGatheringState === 'complete') return;
    await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => finish(new Error('ICE gathering timed out')), 20000);
        const finish = error => {
            clearTimeout(timeout);
            connection.removeEventListener('icegatheringstatechange', changed);
            if (error) reject(error); else resolve();
        };
        const changed = () => { if (connection.iceGatheringState === 'complete') finish(); };
        connection.addEventListener('icegatheringstatechange', changed);
        changed();
    });
}

async function create_answer(connection, config) {
    const answer = await connection.createAnswer();
    const offered_level = connection.remoteDescription.sdp.match(/^a=fmtp:102 [^\r\n]*profile-level-id=42[ce]0([0-9a-f]{2})(?=[;\r\n]|$)/im)?.[1].toLowerCase();
    if (!offered_level) throw new Error('Missing H.264 level in the server offer.');
    if (offered_level === '1f') return answer;
    // Chromium's default codec list advertises level 3.1. Confirm the actual
    // receive configuration before raising the answer's limit to the offered level.
    const capability = await navigator.mediaCapabilities?.decodingInfo?.({
        type: 'webrtc',
        video: {contentType: `video/H264;packetization-mode=1;profile-level-id=42e0${offered_level}`,
            width: config.width, height: config.height, bitrate: config.bitrate, framerate: config.fps},
    });
    if (!capability?.supported) throw new Error('This browser cannot confirm H.264 decoding at the selected resolution, frame rate, and bitrate. Try configuring the server for 1280x720 at 30 FPS and 8 Mbit/s.');
    answer.sdp = answer.sdp.replace(/^(a=fmtp:102 [^\r\n]*profile-level-id=42[ce]0)[0-9a-f]{2}(?=[;\r\n]|$)/gim,
        (_, prefix) => prefix + offered_level);
    return answer;
}

function connect() {
    disconnect();
    if (!RTCRtpReceiver.getCapabilities('video')?.codecs.some(codec => codec.mimeType.toLowerCase() === 'video/h264')) {
        element('message').textContent = 'This browser has no H.264 WebRTC decoder. Enable its H.264 codec or use Edge or Chrome.';
        return;
    }
    let stream_config = null;
    connection_label('Disconnect');
    const ws = socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`);
    ws.onopen = () => ws.send(JSON.stringify({type: 'auth', token}));
    ws.onclose = () => { if (socket === ws) disconnect('Session ended. Click Connect to start a new stream.'); };
    ws.onerror = () => { if (socket === ws) disconnect('Could not connect to the signaling server.'); };
    ws.onmessage = async event => {
        if (socket !== ws) return;
        try {
            const message = JSON.parse(event.data);
            if (message.type === 'error') { disconnect(message.message); return; }
            if (message.type === 'config') {
                stream_config = message;
                element('video').style.setProperty('--stream-width', `${message.width}px`);
                element('video').style.setProperty('--stream-height', `${message.height}px`);
                const connection = peer = new RTCPeerConnection({iceServers: message.iceServers, iceTransportPolicy: message.iceTransportPolicy});
                connection.ondatachannel = event => {
                    if (socket !== ws || event.channel.label !== 'falcor-input-v1') { event.channel.close(); return; }
                    if (remoteInput) remoteInput.close();
                    remoteInput = new window.FalcorInput(event.channel, message);
                };
                connection.ontrack = event => {
                    if (socket !== ws) return;
                    element('video').srcObject = new MediaStream([event.track]);
                    element('video').play().catch(() => {});
                };
                connection.onconnectionstatechange = () => {
                    if (socket !== ws) return;
                    if (['failed', 'disconnected'].includes(connection.connectionState)) disconnect('Media connection lost. Click Connect to retry.');
                };
            } else if (message.type === 'offer') {
                if (!peer) throw new Error('Offer arrived before connection configuration');
                const connection = peer;
                await connection.setRemoteDescription({type: 'offer', sdp: message.sdp});
                await connection.setLocalDescription(await create_answer(connection, stream_config));
                await gather(connection);
                if (socket === ws) ws.send(JSON.stringify({type: 'answer', sdp: connection.localDescription.sdp}));
            } else if (message.type === 'stats') {
                serverStats = message.stats;
                update_statistics();
            }
        } catch (error) { if (socket === ws) disconnect(error.message); }
    };
}

setInterval(async () => {
    const connection = peer;
    if (!connection) return;
    try {
        const stats = await connection.getStats();
        if (peer !== connection) return;
        let candidatePair;
        stats.forEach(report => {
            if (report.type === 'inbound-rtp' && report.kind === 'video') {
                const codec = stats.get(report.codecId);
                browserStats = {framesDecoded: report.framesDecoded || 0, keyFramesDecoded: report.keyFramesDecoded || 0,
                    framesPerSecond: report.framesPerSecond, packetsLost: report.packetsLost, jitter: report.jitter,
                    codec: codec?.mimeType, width: report.frameWidth, height: report.frameHeight};
            }
            if (report.type === 'transport' && report.selectedCandidatePairId) candidatePair = stats.get(report.selectedCandidatePairId);
            if (!candidatePair && report.type === 'candidate-pair' && report.nominated && report.state === 'succeeded') candidatePair = report;
        });
        if (candidatePair) {
            browserStats.localCandidateType = stats.get(candidatePair.localCandidateId)?.candidateType;
            browserStats.remoteCandidateType = stats.get(candidatePair.remoteCandidateId)?.candidateType;
            browserStats.rttMs = (candidatePair.currentRoundTripTime || 0) * 1000;
        }
        update_statistics();
    } catch (error) { /* Closing a connection can interrupt getStats. */ }
}, 500);

function connection_label(label) {
    element('connection-label').textContent = label;
    element('connection').setAttribute('aria-label', label);
    element('connection').title = label;
}

element('sidebar-toggle').addEventListener('click', () => {
    const collapsed = element('sidebar').classList.toggle('collapsed');
    const label = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
    element('sidebar-toggle').setAttribute('aria-expanded', String(!collapsed));
    element('sidebar-toggle').setAttribute('aria-label', label);
    element('sidebar-toggle').title = label;
    element('sidebar-toggle-label').textContent = label;
});
element('fit-view').addEventListener('change', () => {
    element('viewport').classList.toggle('actual-size', !element('fit-view').checked);
});
element('fullscreen').disabled = !document.fullscreenEnabled;
element('fullscreen').addEventListener('click', async () => {
    try {
        if (document.fullscreenElement) await document.exitFullscreen();
        else await element('viewport').requestFullscreen();
    } catch (error) {
        element('message').textContent = `Could not enter or leave full screen: ${error.message}`;
    }
});

element('connection').addEventListener('click', () => socket ? disconnect() : connect());
element('statistics').addEventListener('toggle', update_statistics);
window.addEventListener('pagehide', () => disconnect());
window.falcorStreaming = {connect, disconnect, get stats() { return {browser: browserStats, server: serverStats}; }};
connect();
