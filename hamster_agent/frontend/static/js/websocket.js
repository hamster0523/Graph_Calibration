/**
 * WebSocket模块 - WebSocket连接管理和实时通信
 */

class WebSocketManager {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectInterval = 3000;
        this.isConnecting = false;
        this.listeners = new Map();
        this.heartbeatInterval = null;
        this.pendingPing = false;

        this.connect();

        // 页面关闭时清理连接
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }

    connect() {
        if (this.isConnecting) return;

        this.isConnecting = true;
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        console.log('WebSocket connecting to:', wsUrl);

        try {
            this.ws = new WebSocket(wsUrl);
            this.setupEventHandlers();
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.isConnecting = false;
            this.handleReconnect();
        }
    }

    setupEventHandlers() {
        this.ws.onopen = (event) => {
            console.log('WebSocket connected');
            this.isConnecting = false;
            this.reconnectAttempts = 0;

            this.startHeartbeat();
            this.updateConnectionStatus(true);
            this.emit('connected', { timestamp: new Date().toISOString() });
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket disconnected:', event.code, event.reason);
            this.isConnecting = false;
            this.stopHeartbeat();
            this.updateConnectionStatus(false);
            this.emit('disconnected', { code: event.code, reason: event.reason });

            // 非正常关闭时尝试重连
            if (event.code !== 1000) {
                this.handleReconnect();
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.isConnecting = false;
            this.emit('error', { error });
        };
    }

    handleMessage(data) {
        const { type, data: messageData } = data;

        // 添加调试日志
        console.log('WebSocket received message:', type, messageData);

        // 处理pong响应
        if (type === 'pong') {
            this.pendingPing = false;
            return;
        }

        // 触发相应的监听器
        this.emit(type, messageData);

        // 记录活动日志
        if (type === 'user_message' || type === 'agent_response' || type === 'status_update' || type === 'agent_stream') {
            this.logActivity(type, messageData);
        }
    }

    send(type, data = {}) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const message = { type, data, timestamp: new Date().toISOString() };
            this.ws.send(JSON.stringify(message));
            return true;
        } else {
            console.warn('WebSocket not connected, message not sent:', type);
            return false;
        }
    }

    startHeartbeat() {
        this.stopHeartbeat();
        this.heartbeatInterval = setInterval(() => {
            if (this.pendingPing) {
                // 上次ping还没收到pong，连接可能有问题
                console.warn('WebSocket heartbeat timeout, reconnecting...');
                this.reconnect();
                return;
            }

            if (this.send('ping')) {
                this.pendingPing = true;
            }
        }, 30000); // 30秒心跳
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
        this.pendingPing = false;
    }

    reconnect() {
        this.cleanup();
        setTimeout(() => {
            this.connect();
        }, 1000);
    }

    handleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            this.emit('maxReconnectAttemptsReached');
            return;
        }

        this.reconnectAttempts++;
        console.log(`Reconnecting... Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);

        setTimeout(() => {
            this.connect();
        }, this.reconnectInterval * this.reconnectAttempts);
    }

    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
    }

    off(event, callback) {
        if (this.listeners.has(event)) {
            const callbacks = this.listeners.get(event);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
    }

    emit(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error('WebSocket event callback error:', error);
                }
            });
        }
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connectionStatus');
        const statusDot = statusElement?.querySelector('.status-dot');
        const statusText = statusElement?.querySelector('.status-text');

        if (statusElement) {
            statusElement.className = `connection-indicator ${connected ? 'connected' : 'disconnected'}`;
        }

        if (statusText) {
            statusText.textContent = connected ? 'Connected' :
                this.reconnectAttempts > 0 ? `Reconnecting (${this.reconnectAttempts})...` : 'Disconnected';
        }
    }

    logActivity(type, data) {
        const activityLog = document.getElementById('activityLog');
        if (!activityLog) return;

        const entry = document.createElement('div');
        entry.className = 'log-entry';

        const timestamp = document.createElement('span');
        timestamp.className = 'timestamp';
        timestamp.textContent = new Date().toLocaleTimeString();

        const message = document.createElement('span');
        message.className = 'message';

        switch (type) {
            case 'user_message':
                message.textContent = 'User sent message';
                entry.className += ' user-entry';
                break;
            case 'agent_response':
                message.textContent = 'Agent responded';
                entry.className += ' agent-entry';
                break;
            case 'status_update':
                message.textContent = `Status: ${data.status || 'unknown'}`;
                entry.className += ' status-entry';
                break;
            default:
                message.textContent = `Event: ${type}`;
        }

        entry.appendChild(timestamp);
        entry.appendChild(message);

        // 添加到日志顶部
        activityLog.insertBefore(entry, activityLog.firstChild);

        // 限制日志条目数量
        const entries = activityLog.querySelectorAll('.log-entry');
        if (entries.length > 50) {
            entries[entries.length - 1].remove();
        }
    }

    getConnectionState() {
        if (!this.ws) return 'CLOSED';
        switch (this.ws.readyState) {
            case WebSocket.CONNECTING: return 'CONNECTING';
            case WebSocket.OPEN: return 'OPEN';
            case WebSocket.CLOSING: return 'CLOSING';
            case WebSocket.CLOSED: return 'CLOSED';
            default: return 'UNKNOWN';
        }
    }

    cleanup() {
        this.stopHeartbeat();
        if (this.ws) {
            this.ws.close(1000, 'Page unloading');
            this.ws = null;
        }
    }
}

// 全局WebSocket管理器实例
window.wsManager = new WebSocketManager();
// 同时导出类以供app.js使用
window.WebSocketManager = WebSocketManager;

// 导出给其他模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebSocketManager;
}
