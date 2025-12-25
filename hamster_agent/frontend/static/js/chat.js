/**
 * 聊天模块 - 管理聊天界面和消息处理
 */

class ChatManager {
    constructor() {
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.charCount = document.getElementById('charCount');
        this.clearChatBtn = document.getElementById('clearChatBtn');
        this.resetAgentBtn = document.getElementById('resetAgentBtn');

        this.isProcessing = false;
        this.messageHistory = [];

        // 流式消息相关
        this.currentStreamingMessage = null;
        this.currentStep = 0;
        this.streamingSteps = [];

        this.init();
        this.setupWebSocketListeners();
    }

    init() {
        // 设置事件监听器, 绑定发送按钮点击事件
        this.sendButton?.addEventListener('click', () => this.sendMessage());

        // 输入框事件
        this.messageInput?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        this.messageInput?.addEventListener('input', () => {
            this.updateCharCount();
            this.autoResize();
        });

        // 快捷操作
        this.clearChatBtn?.addEventListener('click', () => this.clearChat());
        this.resetAgentBtn?.addEventListener('click', () => this.resetAgent());

        // 快速建议按钮
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('suggestion-btn')) {
                const message = e.target.getAttribute('data-message');
                if (message) {
                    this.messageInput.value = message;
                    this.sendMessage();
                }
            }
        });

        this.updateCharCount();
    }

    setupWebSocketListeners() {
        // 添加延迟以确保WebSocket管理器已经初始化
        const setupListeners = () => {
            if (!window.wsManager) {
                console.warn('WebSocket manager not available, retrying in 100ms...');
                setTimeout(setupListeners, 100);
                return;
            }

            console.log('Setting up WebSocket listeners...');

            // 用户消息
            window.wsManager.on('user_message', (data) => {
                console.log('📨 User message received:', data);
                this.addMessage(data.message, 'user', data.timestamp);
            });

            // Agent响应（保留兼容性）
            window.wsManager.on('agent_response', (data) => {
                console.log('📨 Agent response received:', data);
                this.addMessage(data.response, 'assistant', data.timestamp);
                this.setProcessing(false);
            });

            // 新增：Agent流式消息处理
            window.wsManager.on('agent_stream', (data) => {
                console.log('📨 Received agent stream message:', data);
                this.handleAgentStream(data);
            });

            // 状态更新
            window.wsManager.on('status_update', (data) => {
                console.log('📨 Status update received:', data);
                this.updateAgentStatus(data);
            });

            // 连接建立
            window.wsManager.on('connection_established', (data) => {
                this.addSystemMessage('WebSocket connection established');
            });

            // 连接断开
            window.wsManager.on('disconnected', () => {
                this.addSystemMessage('Connection lost. Attempting to reconnect...', 'warning');
            });

            // 重连成功
            window.wsManager.on('connected', () => {
                this.addSystemMessage('Connection restored', 'success');
            });

            console.log('✅ WebSocket listeners setup completed');
            console.log('WebSocket connection state:', window.wsManager.getConnectionState());
            console.log('Active connections count:', window.wsManager.active_connections?.length || 0);
        };

        setupListeners();
    }

    async sendMessage() {
        const message = this.messageInput?.value?.trim();
        if (!message || this.isProcessing) return;

        try {
            this.setProcessing(true);
            this.clearInput();

            // 添加加载消息
            this.addSystemMessage('正在处理您的消息，请稍等...', 'info');

            // 通过API发送消息（增加超时提示）
            const startTime = Date.now();
            const response = await window.apiClient.chat.sendMessage({
                message: message,
                timestamp: new Date().toISOString()
            });

            // 移除加载消息
            this.removeLastSystemMessage();

            // 如果WebSocket没有处理，手动添加到界面
            if (!this.messageExists(message, 'user')) {
                this.addMessage(message, 'user');
            }

        } catch (error) {
            console.error('Failed to send message:', error);

            // 移除加载消息
            this.removeLastSystemMessage();

            // 根据错误类型显示不同消息
            if (error.message && error.message.includes('timeout')) {
                this.addSystemMessage(
                    '请求超时，这可能是由于Flow初始化需要时间。请稍后重试，或切换到single_agent模式。',
                    'error'
                );
            } else {
                this.addSystemMessage('Failed to send message. Please try again.', 'error');
            }

            this.setProcessing(false);

            // 恢复输入内容
            this.messageInput.value = message;
        }
    }

    addMessage(content, type, timestamp = null) {
        if (!this.messagesContainer) return;

        // 移除欢迎消息
        const welcomeMessage = this.messagesContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }

        const messageElement = document.createElement('div');
        messageElement.className = `message ${type}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = type === 'user' ?
            '<i class="icon-user"></i>' :
            '<i class="icon-robot"></i>';

        const content_div = document.createElement('div');
        content_div.className = 'message-content';

        const text = document.createElement('div');
        text.className = 'message-text';
        text.innerHTML = this.formatMessageContent(content);

        const meta = document.createElement('div');
        meta.className = 'message-meta';
        meta.textContent = timestamp ?
            new Date(timestamp).toLocaleTimeString() :
            new Date().toLocaleTimeString();

        content_div.appendChild(text);
        content_div.appendChild(meta);

        messageElement.appendChild(avatar);
        messageElement.appendChild(content_div);

        this.messagesContainer.appendChild(messageElement);
        this.scrollToBottom();

        // 添加到历史记录
        this.messageHistory.push({
            type,
            content,
            timestamp: timestamp || new Date().toISOString()
        });
    }

    addSystemMessage(content, level = 'info') {
        if (!this.messagesContainer) return;

        const messageElement = document.createElement('div');
        messageElement.className = `system-message ${level}`;
        messageElement.setAttribute('data-system-message', 'true'); // 添加标识

        const icon = document.createElement('i');
        icon.className = level === 'error' ? 'icon-alert-circle' :
            level === 'warning' ? 'icon-alert-triangle' :
                level === 'success' ? 'icon-check-circle' :
                    'icon-info';

        const text = document.createElement('span');
        text.textContent = content;

        const timestamp = document.createElement('span');
        timestamp.className = 'timestamp';
        timestamp.textContent = new Date().toLocaleTimeString();

        messageElement.appendChild(icon);
        messageElement.appendChild(text);
        messageElement.appendChild(timestamp);

        this.messagesContainer.appendChild(messageElement);
        this.scrollToBottom();
    }

    removeLastSystemMessage() {
        if (!this.messagesContainer) return;

        const systemMessages = this.messagesContainer.querySelectorAll('[data-system-message="true"]');
        if (systemMessages.length > 0) {
            const lastSystemMessage = systemMessages[systemMessages.length - 1];
            lastSystemMessage.remove();
        }
    }

    formatMessageContent(content) {
        // 处理换行
        content = content.replace(/\n/g, '<br>');

        // 处理代码块
        content = content.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        content = content.replace(/`([^`]*)`/g, '<code>$1</code>');

        // 处理链接
        content = content.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>');

        return content;
    }

    setProcessing(processing) {
        this.isProcessing = processing;

        if (this.sendButton) {
            this.sendButton.disabled = processing;
            this.sendButton.classList.toggle('processing', processing);
        }

        if (this.messageInput) {
            this.messageInput.disabled = processing;
        }

        if (processing) {
            this.addTypingIndicator();
        } else {
            this.removeTypingIndicator();
        }
    }

    addTypingIndicator() {
        this.removeTypingIndicator();

        const indicator = document.createElement('div');
        indicator.className = 'message assistant typing-indicator';
        indicator.innerHTML = `
            <div class="message-avatar">
                <i class="icon-robot"></i>
            </div>
            <div class="message-content">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;

        this.messagesContainer?.appendChild(indicator);
        this.scrollToBottom();
    }

    removeTypingIndicator() {
        const indicator = this.messagesContainer?.querySelector('.typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    clearInput() {
        if (this.messageInput) {
            this.messageInput.value = '';
            this.updateCharCount();
            this.autoResize();
        }
    }

    updateCharCount() {
        if (this.messageInput && this.charCount) {
            const count = this.messageInput.value.length;
            this.charCount.textContent = count;
        }
    }

    autoResize() {
        if (this.messageInput) {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
        }
    }

    scrollToBottom() {
        if (this.messagesContainer) {
            setTimeout(() => {
                this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
            }, 100);
        }
    }

    updateAgentStatus(status) {
        // 更新状态显示
        const statusBadge = document.getElementById('agentStatus');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const lastAction = document.getElementById('lastAction');

        if (statusBadge) {
            statusBadge.textContent = status.status || 'Unknown';
            statusBadge.className = `status-badge ${status.status?.toLowerCase() || 'idle'}`;
        }

        if (progressFill && progressText) {
            const progress = status.max_steps > 0 ?
                (status.current_step / status.max_steps) * 100 : 0;
            progressFill.style.width = `${progress}%`;
            progressText.textContent = `${status.current_step || 0} / ${status.max_steps || 20}`;
        }

        if (lastAction) {
            lastAction.textContent = status.last_action || 'Ready';
        }
    }

    messageExists(content, type) {
        const messages = this.messagesContainer?.querySelectorAll(`.message.${type} .message-text`);
        if (!messages) return false;

        return Array.from(messages).some(msg =>
            msg.textContent.trim() === content.trim()
        );
    }

    async clearChat() {
        if (!confirm('Are you sure you want to clear the chat history?')) return;

        // 清除界面消息
        if (this.messagesContainer) {
            this.messagesContainer.innerHTML = `
                <div class="welcome-message">
                    <h2>Welcome to OpenManus</h2>
                    <p>I'm your AI assistant. How can I help you today?</p>
                    <div class="quick-suggestions">
                        <button class="suggestion-btn" data-message="What can you help me with?">
                            What can you help me with?
                        </button>
                        <button class="suggestion-btn" data-message="Analyze the files in my workspace">
                            Analyze my workspace
                        </button>
                        <button class="suggestion-btn" data-message="Help me write some code">
                            Help me write code
                        </button>
                    </div>
                </div>
            `;
        }

        // 清除历史记录
        this.messageHistory = [];

        this.addSystemMessage('Chat history cleared', 'info');
    }

    async resetAgent() {
        try {
            await window.apiClient.agent.reset();
            this.addSystemMessage('Agent has been reset', 'success');
        } catch (error) {
            console.error('Failed to reset agent:', error);
            this.addSystemMessage('Failed to reset agent', 'error');
        }
    }

    // 新增：处理Agent流式消息
    handleAgentStream(streamData) {
        console.log('🎯 Handling agent stream:', streamData);
        const { message_type, data, step, total_steps, timestamp } = streamData;

        switch (message_type) {
            case 'start':
                this.addStreamingStep(`🚀 开始任务: ${data.description}`, 'start', step);
                this.updateProgress(step, total_steps);
                break;

            case 'step_start':
                this.addStreamingStep(`▶️ 步骤 ${step}`, 'step', step, data.description);
                this.updateProgress(step, total_steps);
                break;

            case 'think_start':
                this.addStreamingStep('🤔 思考中...', 'thinking', step);
                break;

            case 'think':
                this.updateStreamingStep('🧠 思考完成', data.content, 'thought', step);
                break;

            case 'act':
                this.addStreamingStep(`🔧 执行工具: ${data.tool_name}`, 'action', step, data.description);
                break;

            case 'observe':
                this.updateStreamingStep(`👁 观察结果`, data.result, 'observation', step, data.success);
                break;

            case 'step_complete':
                this.updateStreamingStep(`✅ 步骤 ${step} 完成`, data.result, 'step-complete', step);
                break;

            case 'complete':
                this.completeStreamingMessage(data.result, timestamp);
                this.setProcessing(false);
                break;

            case 'error':
                this.addStreamingStep(`❌ 错误: ${data.error}`, 'error', step);
                this.setProcessing(false);
                break;
        }

        this.scrollToBottom();
    }

    addStreamingStep(title, type, step, details = null) {
        // 确保有流式消息容器
        this.ensureStreamingMessage();

        const stepElement = document.createElement('div');
        stepElement.className = `stream-step ${type}`;
        stepElement.setAttribute('data-step', step);

        const header = document.createElement('div');
        header.className = 'step-header';
        header.innerHTML = `
            <span class="step-icon">${this.getStepIcon(type)}</span>
            <span class="step-title">${title}</span>
            <span class="step-time">${new Date().toLocaleTimeString()}</span>
        `;

        stepElement.appendChild(header);

        if (details) {
            const content = document.createElement('div');
            content.className = 'step-content';
            content.innerHTML = this.formatStreamContent(details);
            stepElement.appendChild(content);
        }

        // 添加加载动画
        if (type === 'thinking' || type === 'action') {
            stepElement.classList.add('loading');
        }

        this.currentStreamingMessage.appendChild(stepElement);
        this.streamingSteps[step] = stepElement;
    }

    updateStreamingStep(title, content, type, step, success = true) {
        const stepElement = this.streamingSteps[step];
        if (!stepElement) return;

        // 移除加载状态
        stepElement.classList.remove('loading');
        stepElement.classList.add(success ? 'success' : 'error');

        // 更新标题
        const titleElement = stepElement.querySelector('.step-title');
        if (titleElement) {
            titleElement.textContent = title;
        }

        // 添加或更新内容
        let contentElement = stepElement.querySelector('.step-content');
        if (!contentElement) {
            contentElement = document.createElement('div');
            contentElement.className = 'step-content';
            stepElement.appendChild(contentElement);
        }

        contentElement.innerHTML = this.formatStreamContent(content);
    }

    ensureStreamingMessage() {
        if (!this.currentStreamingMessage) {
            // 移除欢迎消息
            const welcomeMessage = this.messagesContainer.querySelector('.welcome-message');
            if (welcomeMessage) {
                welcomeMessage.remove();
            }

            // 移除打字指示器
            this.removeTypingIndicator();

            // 创建流式消息容器
            const messageElement = document.createElement('div');
            messageElement.className = 'message assistant streaming';
            messageElement.innerHTML = `
                <div class="message-avatar">
                    <i class="icon-robot"></i>
                </div>
                <div class="message-content">
                    <div class="streaming-header">
                        <span class="streaming-title">🤖 AI Agent 执行中...</span>
                        <div class="streaming-progress">
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: 0%"></div>
                            </div>
                            <span class="progress-text">0%</span>
                        </div>
                    </div>
                    <div class="streaming-steps"></div>
                </div>
            `;

            this.messagesContainer.appendChild(messageElement);
            this.currentStreamingMessage = messageElement.querySelector('.streaming-steps');
            this.streamingSteps = [];
        }
    }

    completeStreamingMessage(finalResult, timestamp) {
        if (!this.currentStreamingMessage) return;

        const messageElement = this.currentStreamingMessage.closest('.message');

        // 移除流式类
        messageElement.classList.remove('streaming');
        messageElement.classList.add('completed');

        // 添加最终结果
        const finalElement = document.createElement('div');
        finalElement.className = 'final-result';
        finalElement.innerHTML = `
            <div class="result-header">
                <strong>📋 执行结果:</strong>
            </div>
            <div class="result-content">
                ${this.formatMessageContent(finalResult)}
            </div>
        `;

        this.currentStreamingMessage.appendChild(finalElement);

        // 添加时间戳
        const meta = document.createElement('div');
        meta.className = 'message-meta';
        meta.textContent = timestamp ?
            new Date(timestamp).toLocaleTimeString() :
            new Date().toLocaleTimeString();

        messageElement.querySelector('.message-content').appendChild(meta);

        // 重置流式状态
        this.currentStreamingMessage = null;
        this.currentStep = 0;
        this.streamingSteps = [];

        // 添加到历史记录
        this.messageHistory.push({
            type: 'assistant',
            content: finalResult,
            timestamp: timestamp || new Date().toISOString()
        });
    }

    updateProgress(current, total) {
        const progressFill = document.querySelector('.streaming-progress .progress-fill');
        const progressText = document.querySelector('.streaming-progress .progress-text');

        if (progressFill && progressText && total > 0) {
            const percent = Math.round((current / total) * 100);
            progressFill.style.width = `${percent}%`;
            progressText.textContent = `${percent}%`;
        }
    }

    formatStreamContent(content) {
        if (!content) return '';

        // 限制长度并格式化
        let formatted = content.toString();
        if (formatted.length > 200) {
            formatted = formatted.substring(0, 200) + '...';
        }

        // 处理特殊字符
        formatted = formatted.replace(/</g, '&lt;').replace(/>/g, '&gt;');

        // 处理换行
        formatted = formatted.replace(/\n/g, '<br>');

        return formatted;
    }

    getStepIcon(type) {
        const icons = {
            'thinking': '🤔',
            'thought': '💭',
            'action': '⚡',
            'observation': '👁',
            'step': '▶️',
            'step-complete': '✅',
            'error': '❌',
            'success': '✅'
        };
        return icons[type] || '•';
    }
}

// 初始化聊天管理器
document.addEventListener('DOMContentLoaded', () => {
    window.chatManager = new ChatManager();
});

// 导出给其他模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChatManager;
}
