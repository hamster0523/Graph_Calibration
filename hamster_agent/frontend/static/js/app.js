/**
 * 主应用程序入口文件
 * 初始化各个模块并管理应用状态
 */

class OpenManusApp {
    constructor() {
        this.currentView = 'chat';
        this.isInitialized = false;
        this.modules = {};

        this.init();
    }

    async init() {
        try {
            console.log('🚀 Initializing OpenManus Web Interface...');

            // 初始化DOM元素
            this.initializeElements();

            // 设置事件监听器
            this.setupEventListeners();

            // 初始化各个模块
            await this.initializeModules();

            // 设置初始视图
            this.switchView('chat');

            // 标记为已初始化
            this.isInitialized = true;

            console.log('✅ OpenManus Web Interface initialized successfully');

            // 显示成功通知
            this.showNotification('initialized successfully', 'success');

        } catch (error) {
            console.error('❌ Failed to initialize OpenManus:', error);
            this.showNotification('Failed to initialize application', 'error');
        }
    }

    initializeElements() {
        // 导航元素
        this.navItems = document.querySelectorAll('.nav-item');
        this.views = document.querySelectorAll('.view');
        this.connectionStatus = document.getElementById('connectionStatus');
        this.notificationContainer = document.getElementById('notificationContainer');
        this.modal = document.getElementById('modal');
        this.modalClose = document.getElementById('modalClose');
    }

    setupEventListeners() {
        // 导航点击事件
        this.navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                const view = item.getAttribute('data-view');
                if (view) {
                    this.switchView(view);
                }
            });
        });

        // 模态框关闭事件
        if (this.modalClose) {
            this.modalClose.addEventListener('click', () => {
                this.hideModal();
            });
        }

        // 点击模态框外部关闭
        if (this.modal) {
            this.modal.addEventListener('click', (e) => {
                if (e.target === this.modal) {
                    this.hideModal();
                }
            });
        }

        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });

        // 窗口大小变化
        window.addEventListener('resize', () => {
            this.handleResize();
        });

        // 页面卸载清理
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }

    async initializeModules() {
        try {
            console.log('📦 Initializing modules...');

            // 使用全局WebSocket管理器实例
            if (window.wsManager) {
                this.modules.websocket = window.wsManager;
                console.log('Using existing WebSocket manager instance');
            } else if (window.WebSocketManager) {
                this.modules.websocket = new window.WebSocketManager();
                window.wsManager = this.modules.websocket;
                console.log('Created new WebSocket manager instance');
            }

            // 初始化API客户端
            if (window.APIClient) {
                this.modules.api = new window.APIClient();

                // 创建具有正确结构的全局引用
                window.apiClient = {
                    // 直接方法（向后兼容）
                    sendMessage: (data) => {
                        if (typeof data === 'object' && data.message) {
                            return this.modules.api.sendMessage(data.message, data.timestamp);
                        }
                        return this.modules.api.sendMessage(data);
                    },
                    getAgentStatus: this.modules.api.getAgentStatus.bind(this.modules.api),

                    // 分组方法
                    chat: {
                        sendMessage: (data) => {
                            if (typeof data === 'object' && data.message) {
                                return this.modules.api.sendMessage(data.message, data.timestamp);
                            }
                            return this.modules.api.sendMessage(data);
                        },
                        getAgentStatus: this.modules.api.getAgentStatus.bind(this.modules.api)
                    },
                    agent: {
                        reset: this.modules.api.resetAgent ? this.modules.api.resetAgent.bind(this.modules.api) : async () => {
                            return this.modules.api.post('/api/agent/reset');
                        }
                    },
                    workspace: {
                        listFiles: this.modules.api.listFiles ? this.modules.api.listFiles.bind(this.modules.api) : async () => {
                            return this.modules.api.get('/api/workspace/files');
                        },
                        uploadFile: async (file, onProgress) => {
                            // 验证文件对象
                            if (!file || !file.name) {
                                throw new Error('Invalid file object');
                            }

                            // 检查文件大小 (50MB限制)
                            const maxSize = 50 * 1024 * 1024;
                            if (file.size > maxSize) {
                                throw new Error('File size exceeds 50MB limit');
                            }

                            const formData = new FormData();
                            formData.append('file', file);

                            // 使用XMLHttpRequest来支持进度回调和正确的文件上传
                            return new Promise((resolve, reject) => {
                                const xhr = new XMLHttpRequest();

                                // 进度监听
                                if (onProgress) {
                                    xhr.upload.addEventListener('progress', (e) => {
                                        if (e.lengthComputable) {
                                            const percentComplete = (e.loaded / e.total) * 100;
                                            onProgress(percentComplete);
                                        }
                                    });
                                }

                                // 完成监听
                                xhr.addEventListener('load', () => {
                                    if (xhr.status >= 200 && xhr.status < 300) {
                                        try {
                                            const response = JSON.parse(xhr.responseText);
                                            resolve(response);
                                        } catch (e) {
                                            resolve(xhr.responseText);
                                        }
                                    } else {
                                        try {
                                            const errorResponse = JSON.parse(xhr.responseText);
                                            reject(new Error(errorResponse.detail || `HTTP ${xhr.status}`));
                                        } catch (e) {
                                            reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
                                        }
                                    }
                                });

                                // 错误监听
                                xhr.addEventListener('error', () => {
                                    reject(new Error('Network error during upload'));
                                });

                                // 发送请求
                                xhr.open('POST', '/api/workspace/upload');
                                xhr.send(formData);
                            });
                        },
                        deleteFile: this.modules.api.deleteFile ? this.modules.api.deleteFile.bind(this.modules.api) : async (filePath) => {
                            return this.modules.api.delete(`/api/workspace/files/${encodeURIComponent(filePath)}`);
                        }
                    },
                    config: {
                        getConfiguration: this.modules.api.getConfiguration ? this.modules.api.getConfiguration.bind(this.modules.api) : async () => {
                            return this.modules.api.get('/api/config');
                        },
                        updateConfiguration: this.modules.api.updateConfiguration ? this.modules.api.updateConfiguration.bind(this.modules.api) : async (config) => {
                            return this.modules.api.post('/api/config', { body: config });
                        },
                        testConfiguration: this.modules.api.testConfiguration ? this.modules.api.testConfiguration.bind(this.modules.api) : async (config) => {
                            return this.modules.api.post('/api/config/test', { body: config });
                        }
                    }
                };

                console.log('✅ API client initialized with structured interface');
            }

            // 初始化聊天模块
            if (window.ChatModule) {
                this.modules.chat = new window.ChatModule();
            }

            // 初始化工作空间模块
            if (window.WorkspaceModule) {
                this.modules.workspace = new window.WorkspaceModule();
            }

            // 初始化配置模块
            if (window.ConfigModule) {
                this.modules.config = new window.ConfigModule();
            }

            // 初始化Flow Agent Selector模块
            if (window.FlowAgentSelector) {
                this.modules.flowAgentSelector = new window.FlowAgentSelector();
                // 延迟初始化以确保所有DOM元素和API客户端就绪
                setTimeout(() => {
                    if (this.modules.flowAgentSelector.initialize) {
                        console.log('🔧 Initializing FlowAgentSelector with API client available:', !!window.apiClient);
                        this.modules.flowAgentSelector.initialize();
                    }
                    this.setupFlowAgentSelector();
                }, 200); // 增加延迟时间确保API客户端已就绪
            }

            console.log('✅ Modules initialized:', Object.keys(this.modules));

        } catch (error) {
            console.error('❌ Module initialization failed:', error);
            throw error;
        }
    }

    switchView(viewName) {
        try {
            console.log(`🔄 Switching to view: ${viewName}`);

            // 更新当前视图
            this.currentView = viewName;

            // 更新导航状态
            this.navItems.forEach(item => {
                item.classList.remove('active');
                if (item.getAttribute('data-view') === viewName) {
                    item.classList.add('active');
                }
            });

            // 更新视图显示
            this.views.forEach(view => {
                view.classList.remove('active');
                if (view.id === viewName + 'View') {
                    view.classList.add('active');
                }
            });

            // 通知模块视图变化
            this.notifyViewChange(viewName);

            // 更新URL（可选）
            if (history.pushState) {
                const newUrl = `${window.location.pathname}#${viewName}`;
                history.pushState(null, '', newUrl);
            }

        } catch (error) {
            console.error('❌ Failed to switch view:', error);
            this.showNotification('Failed to switch view', 'error');
        }
    }

    notifyViewChange(viewName) {
        // 通知各模块视图变化
        Object.values(this.modules).forEach(module => {
            if (module && typeof module.onViewChange === 'function') {
                try {
                    module.onViewChange(viewName);
                } catch (error) {
                    console.error('Module view change handler failed:', error);
                }
            }
        });
    }

    updateConnectionStatus(status) {
        if (!this.connectionStatus) return;

        // 移除所有状态类
        this.connectionStatus.classList.remove('connected', 'disconnected', 'connecting');

        // 添加新状态类
        this.connectionStatus.classList.add(status);

        // 更新状态文本
        const statusText = this.connectionStatus.querySelector('.status-text');
        if (statusText) {
            const statusMessages = {
                connected: 'Connected',
                disconnected: 'Disconnected',
                connecting: 'Connecting...'
            };
            statusText.textContent = statusMessages[status] || status;
        }
    }

    showNotification(message, type = 'info', duration = 3000) {
        if (!this.notificationContainer) return;

        // 简单的HTML转义函数
        const escapeHtml = (unsafe) => {
            return unsafe
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        };

        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-message">${escapeHtml(message)}</span>
                <button class="notification-close">×</button>
            </div>
        `;

        // 设置关闭事件
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => {
            this.removeNotification(notification);
        });

        // 添加到容器
        this.notificationContainer.appendChild(notification);

        // 自动消失
        if (duration > 0) {
            setTimeout(() => {
                this.removeNotification(notification);
            }, duration);
        }

        // 添加动画
        requestAnimationFrame(() => {
            notification.classList.add('show');
        });
    }

    removeNotification(notification) {
        if (!notification.parentNode) return;

        notification.classList.add('hide');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }

    setupFlowAgentSelector() {
        console.log('🔧 Setting up Flow Agent Selector integration...');

        // 连接主页面的选择器到FlowAgentSelector实例
        const flowSelector = document.getElementById('flowSelector');
        const agentSelector = document.getElementById('agentSelector');
        const applyBtn = document.getElementById('applyModeBtn');
        const testBtn = document.getElementById('test-connection');
        const resetBtn = document.getElementById('reset-config');
        const parametersContainer = document.getElementById('parameters-container');
        const parametersSection = document.getElementById('parameters-section');
        const multiAgentsSection = document.getElementById('multi-agents-section');

        if (!flowSelector || !agentSelector || !this.modules.flowAgentSelector) {
            console.warn('Flow selector elements not found or FlowAgentSelector not initialized');
            return;
        }

        // 设置FlowAgentSelector的目标元素
        this.modules.flowAgentSelector.parametersContainer = parametersContainer;

        // 创建Agent选择grid
        this.createAgentGrid();

        // Flow选择变化时
        flowSelector.addEventListener('change', (e) => {
            const selectedFlow = e.target.value;
            console.log('Flow selector changed to:', selectedFlow);

            if (this.modules.flowAgentSelector) {
                this.modules.flowAgentSelector.selectedFlow = selectedFlow;
                this.modules.flowAgentSelector.updateFlowDescription();
                this.modules.flowAgentSelector.updateParametersSection();
                this.modules.flowAgentSelector.updateMultiAgentSection();
                console.log('Flow sections updated');
            }

            // 控制Flow Parameters和Multi-Agent sections的显示
            this.updateFlowSections(selectedFlow);

            // 特殊处理Game Data Analysis
            if (selectedFlow === 'game_data_analysis') {
                console.log('🎮 Game Data Analysis selected - forcing parameter display');
                const parametersSection = document.getElementById('parameters-section');
                const multiAgentsSection = document.getElementById('multi-agents-section');

                if (parametersSection) {
                    parametersSection.style.display = 'block';
                    console.log('Parameters section forced to show');
                }
                if (multiAgentsSection) {
                    multiAgentsSection.style.display = 'block';
                    console.log('Multi-agents section forced to show');
                }

                // 确保参数界面更新
                setTimeout(() => {
                    if (this.modules.flowAgentSelector) {
                        this.modules.flowAgentSelector.updateParametersSection();
                        this.modules.flowAgentSelector.updateMultiAgentSection();
                    }
                }, 100);
            }

            // 特殊处理Data Analysis Flow
            if (selectedFlow === 'data_analysis_flow') {
                console.log('📊 Data Analysis Flow selected - forcing parameter and multi-agent display');
                const parametersSection = document.getElementById('parameters-section');
                const multiAgentsSection = document.getElementById('multi-agents-section');

                if (parametersSection) {
                    parametersSection.style.display = 'block';
                    console.log('Parameters section forced to show for Data Analysis Flow');
                }
                if (multiAgentsSection) {
                    multiAgentsSection.style.display = 'block';
                    console.log('Multi-agents section forced to show for Data Analysis Flow');
                }

                // 确保参数界面更新
                setTimeout(() => {
                    if (this.modules.flowAgentSelector) {
                        this.modules.flowAgentSelector.updateParametersSection();
                        this.modules.flowAgentSelector.updateMultiAgentSection();
                    }
                }, 100);
            }

            console.log('Flow changed to:', selectedFlow);
        });

        // Agent选择变化时
        agentSelector.addEventListener('change', (e) => {
            this.modules.flowAgentSelector.selectedAgent = e.target.value;
            this.modules.flowAgentSelector.updateAgentDescription();
            console.log('Agent changed to:', e.target.value);
        });

        // 应用配置按钮
        if (applyBtn) {
            applyBtn.addEventListener('click', () => {
                // 更新基本选择
                this.modules.flowAgentSelector.selectedFlow = flowSelector.value;
                this.modules.flowAgentSelector.selectedAgent = agentSelector.value;

                // 收集参数
                this.modules.flowAgentSelector.collectParameters();

                // 更新选中的Agent
                this.modules.flowAgentSelector.updateSelectedAgents();

                // 应用配置
                this.modules.flowAgentSelector.applyConfiguration();
            });
        }

        // 测试连接按钮
        if (testBtn) {
            testBtn.addEventListener('click', () => {
                this.modules.flowAgentSelector.testConnection();
            });
        }

        // 重置按钮
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.modules.flowAgentSelector.resetConfiguration();
                this.updateMainUIFromConfig();
            });
        }

        // 绑定FlowAgentSelector的事件处理器
        this.modules.flowAgentSelector.bindEvents();

        // 初始化可用flows
        this.modules.flowAgentSelector.loadAvailableFlowsFromBackend();

        // 初始化界面状态
        setTimeout(() => {
            this.updateFlowSections(flowSelector.value);
        }, 500);

        // 设置全局引用以供访问
        window.flowAgentSelector = this.modules.flowAgentSelector;

        console.log('✅ Flow Agent Selector integrated with main interface');
    }

    createAgentGrid() {
        const agentsGrid = document.getElementById('agents-grid');
        if (!agentsGrid || !this.modules.flowAgentSelector) return;

        // 生成Agent选择grid - 修正：保留可点击的原生复选框，隐藏装饰性的checkmark
        const agents = this.modules.flowAgentSelector.agents;
        agentsGrid.innerHTML = agents.map(agent => `
            <div class="agent-option">
                <label class="agent-checkbox">
                    <input type="checkbox" id="agent-${agent.id}" value="${agent.id}">
                    <span class="checkmark" style="display: none !important;"></span>
                    <div class="agent-info">
                        <div class="agent-name">${agent.name}</div>
                        <div class="agent-desc">${agent.description}</div>
                    </div>
                </label>
            </div>
        `).join('');

        console.log('Agent grid created with', agents.length, 'agents');
    }

    updateFlowSections(selectedFlow) {
        const parametersSection = document.getElementById('parameters-section');
        const multiAgentsSection = document.getElementById('multi-agents-section');

        console.log('🔧 Updating flow sections for:', selectedFlow);

        // 获取Flow配置
        const flowConfig = this.modules.flowAgentSelector?.availableFlows?.find(f => f.name === selectedFlow);

        if (!flowConfig) {
            console.warn('Flow config not found for:', selectedFlow);
            return;
        }

        // 控制Flow Parameters显示
        if (parametersSection) {
            const hasParameters = flowConfig.parameters && Object.keys(flowConfig.parameters).length > 0;
            parametersSection.style.display = hasParameters ? 'block' : 'none';
            console.log('Parameters section:', hasParameters ? 'shown' : 'hidden');
        }

        // 控制Multi-Agent Selection显示
        if (multiAgentsSection) {
            const supportsMultiAgent = flowConfig.supportMultipleAgents === true;
            multiAgentsSection.style.display = supportsMultiAgent ? 'block' : 'none';
            console.log('Multi-agent section:', supportsMultiAgent ? 'shown' : 'hidden');
        }

        // 特殊处理Game Data Analysis Flow
        if (selectedFlow === 'game_data_analysis') {
            console.log('🎮 Activating Game Data Analysis Flow features');
            if (parametersSection) {
                parametersSection.style.display = 'block';
            }
            if (multiAgentsSection) {
                multiAgentsSection.style.display = 'block';
            }
        }

        // 特殊处理Data Analysis Flow
        if (selectedFlow === 'data_analysis_flow') {
            console.log('📊 Activating Data Analysis Flow features');
            if (parametersSection) {
                parametersSection.style.display = 'block';
            }
            if (multiAgentsSection) {
                multiAgentsSection.style.display = 'block';
            }
        }
    }

    updateMainUIFromConfig() {
        // 从FlowAgentSelector同步状态到主UI
        if (this.modules.flowAgentSelector) {
            const flowSelector = document.getElementById('flowSelector');
            const agentSelector = document.getElementById('agentSelector');

            if (flowSelector) {
                flowSelector.value = this.modules.flowAgentSelector.selectedFlow;
            }
            if (agentSelector) {
                agentSelector.value = this.modules.flowAgentSelector.selectedAgent;
            }
        }
    }

    showModal(title, content, actions = []) {
        if (!this.modal) return;

        const modalTitle = document.getElementById('modalTitle');
        const modalBody = document.getElementById('modalBody');
        const modalFooter = document.getElementById('modalFooter');

        // 设置标题
        if (modalTitle) {
            modalTitle.textContent = title;
        }

        // 设置内容
        if (modalBody) {
            if (typeof content === 'string') {
                modalBody.innerHTML = content;
            } else {
                modalBody.innerHTML = '';
                modalBody.appendChild(content);
            }
        }

        // 设置操作按钮
        if (modalFooter && actions.length > 0) {
            modalFooter.innerHTML = '';
            actions.forEach(action => {
                const button = document.createElement('button');
                button.className = `btn ${action.class || 'btn-secondary'}`;
                button.textContent = action.text;
                button.addEventListener('click', action.handler);
                modalFooter.appendChild(button);
            });
        }

        // 显示模态框
        this.modal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }

    hideModal() {
        if (!this.modal) return;

        this.modal.classList.add('hidden');
        document.body.style.overflow = '';
    }

    handleKeyboardShortcuts(e) {
        // Alt + 数字键快速切换视图
        if (e.altKey) {
            const views = ['chat', 'workspace', 'config', 'about'];
            const key = parseInt(e.key);
            if (key >= 1 && key <= views.length) {
                e.preventDefault();
                this.switchView(views[key - 1]);
                return;
            }
        }

        // Ctrl/Cmd + K 快速聚焦搜索
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            // 实现搜索功能
            return;
        }

        // Escape 关闭模态框
        if (e.key === 'Escape') {
            this.hideModal();
        }
    }

    handleResize() {
        // 处理窗口大小变化
        // 通知各模块
        Object.values(this.modules).forEach(module => {
            if (module && typeof module.onResize === 'function') {
                try {
                    module.onResize();
                } catch (error) {
                    console.error('Module resize handler failed:', error);
                }
            }
        });
    }

    cleanup() {
        console.log('🧹 Cleaning up application...');

        // 清理各模块
        Object.values(this.modules).forEach(module => {
            if (module && typeof module.cleanup === 'function') {
                try {
                    module.cleanup();
                } catch (error) {
                    console.error('Module cleanup failed:', error);
                }
            }
        });

        this.modules = {};
        this.isInitialized = false;
    }

    // 公共API方法
    getModule(name) {
        return this.modules[name];
    }

    getCurrentView() {
        return this.currentView;
    }

    isReady() {
        return this.isInitialized;
    }
}

// 等待DOM加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    console.log('🌍 DOM loaded, starting OpenManus...');

    // 创建全局应用实例
    window.OpenManusApp = new OpenManusApp();

    // 处理URL hash
    const hash = window.location.hash.slice(1);
    if (hash && ['chat', 'workspace', 'config', 'about'].includes(hash)) {
        setTimeout(() => {
            window.OpenManusApp.switchView(hash);
        }, 100);
    }
});

// 全局错误处理
window.addEventListener('error', (e) => {
    console.error('🚨 Global error:', e.error);

    if (window.OpenManusApp) {
        window.OpenManusApp.showNotification(
            'An unexpected error occurred',
            'error'
        );
    }
});

// 未处理的Promise错误
window.addEventListener('unhandledrejection', (e) => {
    console.error('🚨 Unhandled promise rejection:', e.reason);

    if (window.OpenManusApp) {
        window.OpenManusApp.showNotification(
            'An unexpected error occurred',
            'error'
        );
    }
});
