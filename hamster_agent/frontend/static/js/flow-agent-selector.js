/**
 * Flow和Agent选择器模块
 * 处理工作流程和代理的选择、配置和切换
 */

class FlowAgentSelector {
    constructor() {
        // 简化初始化，避免复杂的方法调用
        this.agents = [
            { id: 'Manus', name: 'Manus Agent', description: 'Default agent for general tasks' },
            { id: 'DataAnalysisExpert', name: 'Data Analysis Expert', description: 'Specialized in data analysis and reporting' },
            { id: 'ExcelCleanAgent', name: 'Excel Clean Agent', description: 'Cleans and preprocesses Excel files' },
            { id: 'GameDataAnalysisAgent', name: 'Game Data Analysis Agent', description: 'Specialized in game performance data analysis' },
            { id: 'SWEAgent', name: 'SWE Agent', description: 'Software engineering and code analysis agent' },
            { id: 'BrowserAgent', name: 'Browser Agent', description: 'Web browsing and automation agent' },
            { id: 'AnalysisResultQnAAgent', name: 'Analysis Result Q&A Agent', description: 'Specialized in answering questions and providing explanations' },
            { id: 'DataAnalysis', name: 'Data Analysis Agent', description: 'General data analysis capabilities' },
            // Game Data Analysis 专用Agent
            { id: 'MultiDataAnalysisCoordinator', name: 'Multi Data Analysis Coordinator', description: 'Coordinates multiple data analysis processes' },
            { id: 'KeyMetricAnalysisAgent', name: 'Key Metric Analysis Agent', description: 'Analyzes key performance indicators and metrics' }
        ];

        this.availableFlows = [
            {
                name: 'single_agent',
                label: 'Single Agent Flow',
                description: 'Uses a single agent to complete tasks',
                parameters: {},
                supportMultipleAgents: false
            },
            {
                name: 'planning',
                label: 'Planning Flow',
                description: 'Multi-step planning workflow with multiple agents',
                parameters: {},
                supportMultipleAgents: true,
                recommendedAgents: ['Manus', 'DataAnalysisExpert', 'SWEAgent']
            },
            {
                name: 'game_data_analysis',
                label: 'Game Data Analysis',
                description: 'Specialized workflow for game performance data analysis with multiple specialized agents',
                parameters: {
                    data_file_path: {
                        type: 'file',
                        label: 'Data File Path',
                        description: 'Path to the data file (CSV, Excel, etc.) to be analyzed',
                        required: true,
                        placeholder: 'total_data.csv',
                        accept: '.csv,.xlsx,.xls,.json'
                    },
                    new_version_like: {
                        type: 'text',
                        label: 'New Version Identifier',
                        description: 'String pattern to identify which records should be considered as new version',
                        required: true,
                        placeholder: 'e.g., v2024, beta, latest',
                        default: '52.03'
                    }
                },
                supportMultipleAgents: true,
                recommendedAgents: ['ExcelCleanAgent', 'MultiDataAnalysisCoordinator', 'KeyMetricAnalysisAgent', 'AnalysisResultQnAAgent'],
                defaultAgents: ['ExcelCleanAgent', 'MultiDataAnalysisCoordinator', 'KeyMetricAnalysisAgent', 'AnalysisResultQnAAgent']
            },
            {
                name: 'data_analysis_flow',
                label: 'Data Analysis Flow',
                description: 'General data analysis workflow for various data types',
                parameters: {
                    data_file_path: {
                        type: 'file',
                        label: 'Data File Path',
                        description: 'Path to the data file (CSV, Excel, etc.)',
                        required: true,
                        placeholder: 'Select data file (.csv, .xlsx, .json, etc.)',
                        accept: '.csv,.xlsx,.xls,.json,.txt'
                    },
                    new_version_like: {
                        type: 'text',
                        label: 'New Version Identifier',
                        description: 'String pattern to identify which records should be considered as new version',
                        required: true,
                        placeholder: 'e.g., v2024, beta, latest',
                        default: '52.03'
                    }
                },
                supportMultipleAgents: true,
                recommendedAgents: ['Manus', 'DataAnalysisExpert', 'ExcelCleanAgent', 'KeyMetricAnalysisAgent'],
                defaultAgents: ['DataAnalysisExpert', 'ExcelCleanAgent']
            }
        ];

        this.selectedFlow = 'single_agent';
        this.selectedAgent = 'Manus';
        this.selectedAgents = []; // 新增：存储选中的多个Agent
        this.flowParameters = {};
        this.parametersContainer = null; // 外部参数容器

        // 直接初始化，不调用复杂方法
        console.log('FlowAgentSelector constructed successfully');
    }

    // 初始化方法 - 延迟调用
    initialize() {
        this.initializeElements();
        this.loadAvailableFlowsFromBackend();
        console.log('FlowAgentSelector initialized');
    }

    initializeElements() {
        // 初始化DOM元素引用（可以为空，因为我们主要通过外部控制）
        this.flowSelector = document.getElementById('flowSelector');
        this.agentSelector = document.getElementById('agentSelector');
        this.applyBtn = document.getElementById('applyModeBtn');

        console.log('FlowAgentSelector elements initialized:', {
            flowSelector: !!this.flowSelector,
            agentSelector: !!this.agentSelector,
            applyBtn: !!this.applyBtn
        });
    }

    setupEventListeners() {
        // 事件监听器将在app.js中设置，这里保持空实现
        console.log('FlowAgentSelector event listeners setup (managed externally)');
    }

    updateUI() {
        // UI更新将通过外部调用进行
        console.log('FlowAgentSelector UI update');
    }

    async loadAvailableFlowsFromBackend() {
        try {
            console.log('🔧 Loading flows and agents from backend...');

            // 从后端获取可用的flows（包含参数信息）
            const response = await fetch('/api/available-flows');
            if (response.ok) {
                const data = await response.json();
                if (data && data.success && data.flows) {
                    // 转换后端格式到前端格式
                    this.availableFlows = data.flows.map(flow => ({
                        name: flow.id,
                        label: flow.name,
                        description: flow.description,
                        parameters: this.convertParametersFormat(flow.parameters || [])
                    }));
                    console.log('✅ Loaded flows from backend:', this.availableFlows);
                } else {
                    console.warn('Invalid response format from backend');
                }
            } else {
                console.warn(`Failed to fetch flows: ${response.status} ${response.statusText}`);
            }
        } catch (error) {
            console.warn('Failed to load flows from backend, using default list:', error);
            // 如果失败，继续使用默认的flow列表
        }

        try {
            // 从后端获取可用的agents
            const agentResponse = await fetch('/api/available-agents');
            if (agentResponse.ok) {
                const agentData = await agentResponse.json();
                if (agentData && agentData.success && agentData.agents) {
                    this.agents = agentData.agents;
                    console.log('✅ Loaded agents from backend:', this.agents);
                }
            }
        } catch (error) {
            console.warn('Failed to load agents from backend:', error);
        }

        // 确保UI更新
        this.updateParametersSection();
    }

    // 转换后端参数格式到前端格式
    convertParametersFormat(backendParams) {
        const frontendParams = {};
        backendParams.forEach(param => {
            frontendParams[param.name] = {
                type: param.type,
                label: param.label,
                description: param.description,
                required: param.required,
                placeholder: param.placeholder,
                default: param.default,
                accept: param.accept
            };
        });
        return frontendParams;
    }

    updateFlowSelector() {
        if (this.flowSelector) {
            this.flowSelector.innerHTML = '';
            this.flows.forEach(flow => {
                const option = document.createElement('option');
                option.value = flow.id;
                option.textContent = flow.name;
                option.title = flow.description;
                this.flowSelector.appendChild(option);
            });

            // 设置当前选中的flow
            this.flowSelector.value = this.currentConfig.mode;
        }
    }

    updateAgentSelector() {
        if (this.agentSelector) {
            this.agentSelector.innerHTML = '';
            this.availableAgents.forEach(agent => {
                const option = document.createElement('option');
                option.value = agent.id;
                option.textContent = agent.name;
                option.title = agent.description;
                this.agentSelector.appendChild(option);
            });

            // 设置当前选中的agent
            this.agentSelector.value = this.currentConfig.primaryAgent;
        }
    }

    createSelector() {
        const container = document.getElementById('flow-agent-selector');
        if (!container) return;

        container.innerHTML = `
            <div class="selector-container">
                <div class="selector-header">
                    <h3>🔧 Flow & Agent Configuration</h3>
                    <div class="status-indicator" id="config-status">
                        <span class="status-dot"></span>
                        <span class="status-text">Not Configured</span>
                    </div>
                </div>

                <!-- Flow Selection -->
                <div class="config-section">
                    <label class="section-label">
                        <i class="icon">🌊</i>
                        Flow Type
                    </label>
                    <select id="flow-selector" class="styled-select">
                        ${this.availableFlows.map(flow =>
            `<option value="${flow.name}">${flow.label}</option>`
        ).join('')}
                    </select>
                    <div class="description" id="flow-description">
                        ${this.availableFlows[0].description}
                    </div>
                </div>

                <!-- Agent Selection -->
                <div class="config-section">
                    <label class="section-label">
                        <i class="icon">🤖</i>
                        Primary Agent
                    </label>
                    <select id="agent-selector" class="styled-select">
                        ${this.agents.map(agent =>
            `<option value="${agent.id}">${agent.name}</option>`
        ).join('')}
                    </select>
                    <div class="description" id="agent-description">
                        ${this.agents[0].description}
                    </div>
                </div>

                <!-- Multiple Agents Selection (for multi-agent flows) -->
                <div class="config-section" id="multi-agents-section" style="display: none;">
                    <label class="section-label">
                        <i class="icon">👥</i>
                        Additional Agents
                        <span class="hint">(Select agents to work together in this flow)</span>
                    </label>
                    <div class="agents-grid" id="agents-grid">
                        ${this.agents.map(agent => `
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
                        `).join('')}
                    </div>
                    <div class="recommended-agents" id="recommended-agents" style="display: none;">
                        <span class="recommendation-label">💡 Recommended for this flow:</span>
                        <div class="recommendation-buttons" id="recommendation-buttons"></div>
                    </div>
                </div>

                <!-- Dynamic Parameters Section -->
                <div class="config-section" id="parameters-section" style="display: none;">
                    <label class="section-label">
                        <i class="icon">⚙️</i>
                        Flow Parameters
                    </label>
                    <div id="parameters-container">
                        <!-- Dynamic parameters will be inserted here -->
                    </div>
                </div>

                <!-- Action Buttons -->
                <div class="action-buttons">
                    <button id="apply-config" class="btn btn-primary">
                        <i class="icon">✅</i>
                        Apply Configuration
                    </button>
                    <button id="test-connection" class="btn btn-secondary">
                        <i class="icon">🔍</i>
                        Test Connection
                    </button>
                    <button id="reset-config" class="btn btn-danger">
                        <i class="icon">🔄</i>
                        Reset
                    </button>
                </div>

                <!-- Configuration Log -->
                <div class="config-log" id="config-log">
                    <div class="log-header">📋 Configuration Log</div>
                    <div class="log-content" id="log-content">
                        <div class="log-entry">Ready for configuration...</div>
                    </div>
                </div>
            </div>
        `;

        this.updateParametersSection();
        this.updateMultiAgentSection(); // 确保多Agent区域也被更新
    }

    // updateParametersSection() {
    //     console.log('🔧 updateParametersSection called');
    //     console.log('Selected flow:', this.selectedFlow);

    //     // 优先使用外部指定的容器，否则使用内部容器
    //     let parametersContainer = this.parametersContainer || document.getElementById('parameters-container');

    //     if (!parametersContainer) {
    //         console.warn('Parameters container not found');
    //         // 尝试等待一下DOM加载
    //         setTimeout(() => {
    //             parametersContainer = document.getElementById('parameters-container');
    //             if (parametersContainer) {
    //                 this.updateParametersSection();
    //             }
    //         }, 100);
    //         return;
    //     }

    //     // 特殊处理Game Data Analysis - 强制显示参数
    //     if (this.selectedFlow === 'game_data_analysis') {
    //         console.log('🎮 Game Data Analysis: Force showing parameters');
    //         const gameDataParams = {
    //             data_file_path: {
    //                 type: 'file',
    //                 label: 'Data File Path',
    //                 description: 'Path to the data file (CSV, Excel, etc.) to be analyzed',
    //                 required: true,
    //                 placeholder: 'total_data.csv',
    //                 accept: '.csv,.xlsx,.xls,.json'
    //             },
    //             new_version_like: {
    //                 type: 'text',
    //                 label: 'New Version Identifier',
    //                 description: 'String pattern to identify which records should be considered as new version',
    //                 required: true,
    //                 placeholder: 'e.g., v2024, beta, latest',
    //                 default: '52.03'
    //             }
    //         };

    //         let html = '';
    //         Object.entries(gameDataParams).forEach(([paramName, param]) => {
    //             const currentValue = this.flowParameters[paramName] || param.default || '';
    //             html += `
    //                 <div class="parameter-item">
    //                     <label for="param-${paramName}" class="parameter-label">
    //                         ${param.label}
    //                         ${param.required ? '<span class="required">*</span>' : ''}
    //                     </label>
    //                     <div class="parameter-description">${param.description}</div>
    //                     ${this.createParameterInput({ ...param, name: paramName }, currentValue)}
    //                 </div>
    //             `;
    //         });

    //         console.log('Generated Game Data Analysis parameters HTML:', html);
    //         parametersContainer.innerHTML = html;

    //         // 显示参数部分
    //         const parametersSection = document.getElementById('parameters-section');
    //         if (parametersSection) {
    //             parametersSection.style.display = 'block';
    //             console.log('Game Data Analysis parameters section shown');
    //         }

    //         // 绑定事件
    //         setTimeout(() => {
    //             document.querySelectorAll('.parameter-input').forEach(input => {
    //                 input.addEventListener('input', () => this.collectParameters());
    //                 input.addEventListener('change', () => this.collectParameters());
    //             });
    //         }, 100);
    //         return;
    //     }

    //     // 特殊处理Data Analysis Flow - 强制显示参数
    //     if (this.selectedFlow === 'data_analysis_flow') {
    //         console.log('📊 Data Analysis Flow: Force showing parameters');
    //         const dataAnalysisParams = {
    //             data_file_path: {
    //                 type: 'file',
    //                 label: 'Data File Path',
    //                 description: 'Path to the data file (CSV, Excel, etc.)',
    //                 required: true,
    //                 placeholder: 'Select data file (.csv, .xlsx, .json, etc.)',
    //                 accept: '.csv,.xlsx,.xls,.json,.txt'
    //             }
    //         };

    //         let html = '';
    //         Object.entries(dataAnalysisParams).forEach(([paramName, param]) => {
    //             const currentValue = this.flowParameters[paramName] || param.default || '';
    //             html += `
    //                 <div class="parameter-item">
    //                     <label for="param-${paramName}" class="parameter-label">
    //                         ${param.label}
    //                         ${param.required ? '<span class="required">*</span>' : ''}
    //                     </label>
    //                     <div class="parameter-description">${param.description}</div>
    //                     ${this.createParameterInput({ ...param, name: paramName }, currentValue)}
    //                 </div>
    //             `;
    //         });

    //         console.log('Generated Data Analysis parameters HTML:', html);
    //         parametersContainer.innerHTML = html;

    //         // 显示参数部分
    //         const parametersSection = document.getElementById('parameters-section');
    //         if (parametersSection) {
    //             parametersSection.style.display = 'block';
    //             console.log('Data Analysis parameters section shown');
    //         }

    //         // 绑定事件
    //         setTimeout(() => {
    //             document.querySelectorAll('.parameter-input').forEach(input => {
    //                 input.addEventListener('input', () => this.collectParameters());
    //                 input.addEventListener('change', () => this.collectParameters());
    //             });
    //         }, 100);
    //         return;
    //     }

    //     // 找到当前选择的flow
    //     const currentFlow = this.availableFlows?.find(flow => flow.name === this.selectedFlow);
    //     console.log('Current flow found:', currentFlow);

    //     if (!currentFlow || !currentFlow.parameters || Object.keys(currentFlow.parameters).length === 0) {
    //         console.log('No parameters for current flow');
    //         parametersContainer.innerHTML = '<div class="parameter-item"><em>此 Flow 无需额外参数</em></div>';
    //         const parametersSection = document.getElementById('parameters-section');
    //         if (parametersSection) {
    //             parametersSection.style.display = 'none';
    //         }
    //         return;
    //     }

    //     console.log('Flow parameters:', currentFlow.parameters);

    //     // 生成参数输入HTML
    //     let html = '';
    //     Object.entries(currentFlow.parameters).forEach(([paramName, paramDef]) => {
    //         const value = this.flowParameters[paramName] || paramDef.default || '';

    //         html += `
    //             <div class="parameter-item">
    //                 <label class="parameter-label ${paramDef.required ? 'required' : ''}">${paramDef.label || paramName}${paramDef.required ? ' *' : ''}:</label>
    //                 ${this.createParameterInput({ ...paramDef, name: paramName }, value)}
    //                 ${paramDef.description ? `<div class="parameter-description">${paramDef.description}</div>` : ''}
    //             </div>
    //         `;
    //     });

    //     console.log('Generated parameters HTML:', html);
    //     parametersContainer.innerHTML = html;

    //     // 显示参数部分
    //     const parametersSection = document.getElementById('parameters-section');
    //     if (parametersSection) {
    //         parametersSection.style.display = 'block';
    //         console.log('Parameters section shown');
    //     }

    //     // 特殊处理Game Data Analysis
    //     if (this.selectedFlow === 'game_data_analysis') {
    //         console.log('🎮 Special handling for Game Data Analysis Flow');
    //         if (parametersSection) {
    //             parametersSection.style.display = 'block';
    //         }
    //         // 确保参数收集事件绑定
    //         setTimeout(() => {
    //             document.querySelectorAll('.parameter-input').forEach(input => {
    //                 input.addEventListener('input', () => this.collectParameters());
    //                 input.addEventListener('change', () => this.collectParameters());
    //             });
    //         }, 100);
    //     }
    // }
    updateParametersSection() {
        console.log('🔧 updateParametersSection called');
        console.log('Selected flow:', this.selectedFlow);

        // 获取容器（外部指定或 DOM）
        let parametersContainer = this.parametersContainer || document.getElementById('parameters-container');

        if (!parametersContainer) {
            console.warn('Parameters container not found');
            setTimeout(() => {
                parametersContainer = document.getElementById('parameters-container');
                if (parametersContainer) {
                    this.updateParametersSection();
                }
            }, 100);
            return;
        }

        // 特殊流程参数定义
        const specialFlows = {
            game_data_analysis: {
                data_file_path: {
                    type: 'file',
                    label: 'Data File Path',
                    description: 'Path to the data file (CSV, Excel, etc.) to be analyzed',
                    required: true,
                    placeholder: 'total_data.csv',
                    accept: '.csv,.xlsx,.xls,.json'
                },
                new_version_like: {
                    type: 'text',
                    label: 'New Version Identifier',
                    description: 'String pattern to identify which records should be considered as new version',
                    required: true,
                    placeholder: 'e.g., v2024, beta, latest',
                    default: '52.03'
                }
            },
            data_analysis_flow: {
                data_file_path: {
                    type: 'file',
                    label: 'Data File Path',
                    description: 'Path to the data file (CSV, Excel, etc.)',
                    required: true,
                    placeholder: 'Select data file (.csv, .xlsx, .json, etc.)',
                    accept: '.csv,.xlsx,.xls,.json,.txt'
                },
                new_version_like: {
                    type: 'text',
                    label: 'New Version Identifier',
                    description: 'String pattern to identify which records should be considered as new version',
                    required: true,
                    placeholder: 'e.g., v2024, beta, latest',
                    default: '52.03'
                }
            }
        };

        // 如果是特殊流程，使用自定义参数定义
        if (specialFlows[this.selectedFlow]) {
            console.log(`✨ Special flow "${this.selectedFlow}" detected`);
            this.renderParameters(specialFlows[this.selectedFlow]);
            this.bindParameterInputEvents();
            return;
        }

        // 查找当前流程的参数
        const currentFlow = this.availableFlows?.find(flow => flow.name === this.selectedFlow);
        console.log('Current flow found:', currentFlow);

        if (!currentFlow || !currentFlow.parameters || Object.keys(currentFlow.parameters).length === 0) {
            console.log('No parameters for current flow');
            parametersContainer.innerHTML = '<div class="parameter-item"><em>此 Flow 无需额外参数</em></div>';
            const parametersSection = document.getElementById('parameters-section');
            if (parametersSection) {
                parametersSection.style.display = 'none';
            }
            return;
        }

        // 渲染普通流程的参数
        this.renderParameters(currentFlow.parameters);
        this.bindParameterInputEvents();
    }

    renderParameters(paramDefs) {
        let html = '';

        Object.entries(paramDefs).forEach(([paramName, param]) => {
            const currentValue = this.flowParameters[paramName] || param.default || '';
            html += `
                <div class="parameter-item">
                    <label for="param-${paramName}" class="parameter-label">
                        ${param.label}
                        ${param.required ? '<span class="required">*</span>' : ''}
                    </label>
                    <div class="parameter-description">${param.description}</div>
                    ${this.createParameterInput({ ...param, name: paramName }, currentValue)}
                </div>
            `;
        });

        const parametersContainer = this.parametersContainer || document.getElementById('parameters-container');
        if (parametersContainer) {
            parametersContainer.innerHTML = html;
        }

        const parametersSection = document.getElementById('parameters-section');
        if (parametersSection) {
            parametersSection.style.display = 'block';
        }
    }

    bindParameterInputEvents() {
        setTimeout(() => {
            document.querySelectorAll('.parameter-input').forEach(input => {
                input.addEventListener('input', () => this.collectParameters());
                input.addEventListener('change', () => this.collectParameters());
            });
        }, 100);
    }

    createParameterInput(param, value) {
        switch (param.type) {
            case 'file':
                return `
                    <div class="file-input-container">
                        <input
                            type="text"
                            id="param-${param.name}"
                            class="parameter-input file-input"
                            placeholder="${param.placeholder || '请选择文件路径'}"
                            value="${value || ''}"
                            ${param.required ? 'required' : ''}
                        >
                        <button
                            type="button"
                            class="browse-file-btn"
                            data-param-name="${param.name}"
                        >
                            浏览
                        </button>
                    </div>
                `;
            case 'select':
                let options = '';
                if (param.options) {
                    param.options.forEach(option => {
                        const optionValue = option.value || option;
                        const optionLabel = option.label || option;
                        const selected = value === optionValue ? 'selected' : '';
                        options += `<option value="${optionValue}" ${selected}>${optionLabel}</option>`;
                    });
                }
                return `
                    <select
                        id="param-${param.name}"
                        class="parameter-input"
                        ${param.required ? 'required' : ''}
                    >
                        ${options}
                    </select>
                `;
            case 'text':
            default:
                return `
                    <input
                        type="text"
                        id="param-${param.name}"
                        class="parameter-input"
                        placeholder="${param.placeholder || ''}"
                        value="${value || ''}"
                        ${param.required ? 'required' : ''}
                    >
                `;
        }
    }

    browseFile(paramName) {
        // Create a file input element
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = '.csv,.xlsx,.xls,.json,.txt,.data';
        fileInput.style.display = 'none';

        fileInput.onchange = (event) => {
            const file = event.target.files[0];
            if (file) {
                // Get the parameter input element
                const paramInput = document.getElementById(`param-${paramName}`);
                if (paramInput) {
                    // For web applications, we typically work with file objects
                    // In a real scenario, you might upload the file to server first
                    paramInput.value = file.name;
                    this.flowParameters[paramName] = file.name;
                    this.logMessage(`文件已选择: ${file.name}`, 'info');

                    // Trigger parameter collection
                    this.collectParameters();
                }
            }
            // Clean up
            document.body.removeChild(fileInput);
        };

        // Add to DOM and trigger click
        document.body.appendChild(fileInput);
        fileInput.click();
    }

    // 收集所有参数值
    collectParameters() {
        this.flowParameters = {};

        // 特殊处理Game Data Analysis - 直接收集硬编码的参数
        if (this.selectedFlow === 'game_data_analysis') {
            console.log('🎮 Collecting Game Data Analysis parameters');

            const dataFilePathInput = document.getElementById('param-data_file_path');
            const newVersionLikeInput = document.getElementById('param-new_version_like');

            if (dataFilePathInput && dataFilePathInput.value.trim()) {
                this.flowParameters.data_file_path = dataFilePathInput.value.trim();
                console.log('Collected data_file_path:', this.flowParameters.data_file_path);
            }

            if (newVersionLikeInput && newVersionLikeInput.value.trim()) {
                this.flowParameters.new_version_like = newVersionLikeInput.value.trim();
                console.log('Collected new_version_like:', this.flowParameters.new_version_like);
            }

            console.log('Game Data Analysis parameters collected:', this.flowParameters);
            return;
        }

        // 特殊处理Data Analysis Flow - 直接收集硬编码的参数
        if (this.selectedFlow === 'data_analysis_flow') {
            console.log('📊 Collecting Data Analysis Flow parameters');

            const dataFilePathInput = document.getElementById('param-data_file_path');
            if (dataFilePathInput && dataFilePathInput.value.trim()) {
                this.flowParameters.data_file_path = dataFilePathInput.value.trim();
                console.log('Collected data_file_path:', this.flowParameters.data_file_path);
            }

            console.log('Data Analysis Flow parameters collected:', this.flowParameters);
            return;
        }

        // 获取当前flow的参数定义（常规流程）
        const currentFlow = this.availableFlows?.find(flow => flow.name === this.selectedFlow);
        if (!currentFlow || !currentFlow.parameters) {
            console.log('No parameters to collect for current flow');
            return;
        }

        // 遍历每个参数，收集其值
        Object.keys(currentFlow.parameters).forEach(paramName => {
            const paramInput = document.getElementById(`param-${paramName}`);
            if (paramInput) {
                const value = paramInput.value.trim();
                if (value) {
                    this.flowParameters[paramName] = value;
                }
            } else {
                console.warn(`Parameter input not found: param-${paramName}`);
            }
        });

        console.log('Collected parameters:', this.flowParameters);
        this.logMessage(`参数已收集: ${JSON.stringify(this.flowParameters)}`, 'debug');
    }

    bindEvents() {
        // Flow selector change
        const flowSelector = document.getElementById('flow-selector');
        if (flowSelector) {
            flowSelector.addEventListener('change', (e) => {
                this.selectedFlow = e.target.value;
                this.updateFlowDescription();
                this.updateParametersSection();
                this.updateMultiAgentSection(); // 新增：更新多Agent选择区域
                this.logMessage(`Flow changed to: ${this.getFlowName(this.selectedFlow)}`, 'info');
            });
        }

        // Agent selector change
        const agentSelector = document.getElementById('agent-selector');
        if (agentSelector) {
            agentSelector.addEventListener('change', (e) => {
                this.selectedAgent = e.target.value;
                this.updateAgentDescription();
                this.logMessage(`Agent changed to: ${this.getAgentName(this.selectedAgent)}`, 'info');
            });
        }

        // Multi-agent checkboxes change
        document.addEventListener('change', (e) => {
            if (e.target.type === 'checkbox' && e.target.id.startsWith('agent-')) {
                this.updateSelectedAgents();
            }
        });

        // Parameter inputs change
        document.addEventListener('input', (e) => {
            if (e.target.classList.contains('parameter-input')) {
                this.collectParameters();
                const paramName = e.target.id.replace('param-', '');
                this.logMessage(`参数 ${paramName} 已更新: ${e.target.value}`, 'info');
            }
        });

        // Parameter selects change
        document.addEventListener('change', (e) => {
            if (e.target.classList.contains('parameter-input')) {
                this.collectParameters();
                const paramName = e.target.id.replace('param-', '');
                this.logMessage(`参数 ${paramName} 已更新: ${e.target.value}`, 'info');
            }
        });

        // Apply configuration button
        const applyBtn = document.getElementById('apply-config');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => this.applyConfiguration());
        }

        // Test connection button
        const testBtn = document.getElementById('test-connection');
        if (testBtn) {
            testBtn.addEventListener('click', () => this.testConnection());
        }

        // Reset button
        const resetBtn = document.getElementById('reset-config');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetConfiguration());
        }

        // File browse buttons - 使用事件委托
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('browse-file-btn')) {
                const paramName = e.target.getAttribute('data-param-name');
                if (paramName) {
                    this.browseFile(paramName);
                }
            }
        });
    }

    updateFlowDescription() {
        const flow = this.availableFlows.find(f => f.name === this.selectedFlow);
        const descElement = document.getElementById('flow-description');
        if (descElement && flow) {
            descElement.textContent = flow.description;
        }
    }

    updateAgentDescription() {
        const agent = this.agents.find(a => a.id === this.selectedAgent);
        const descElement = document.getElementById('agent-description');
        if (descElement && agent) {
            descElement.textContent = agent.description;
        }
    }

    async applyConfiguration() {
        try {
            this.logMessage('Applying configuration...', 'info');

            // 首先收集当前参数和选中的Agent
            this.collectParameters();
            this.updateSelectedAgents();

            // 调试日志
            console.log('DEBUG - Flow parameters:', this.flowParameters);
            console.log('DEBUG - Selected agents:', this.selectedAgents);
            this.logMessage(`DEBUG - Parameters: ${JSON.stringify(this.flowParameters)}`, 'info');
            this.logMessage(`DEBUG - Selected agents: [${this.selectedAgents.join(', ')}]`, 'info');

            // 验证必需参数
            if (this.selectedFlow === 'game_data_analysis') {
                // Game Data Analysis 特殊验证
                if (!this.flowParameters.data_file_path) {
                    throw new Error('Required parameter "Data File Path" is missing');
                }
                if (!this.flowParameters.new_version_like) {
                    throw new Error('Required parameter "New Version Identifier" is missing');
                }
            } else if (this.selectedFlow === 'data_analysis_flow') {
                // Data Analysis Flow 特殊验证
                if (!this.flowParameters.data_file_path) {
                    throw new Error('Required parameter "Data File Path" is missing');
                }
            } else {
                // 常规验证
                const currentFlow = this.availableFlows.find(f => f.name === this.selectedFlow);
                if (currentFlow && currentFlow.parameters) {
                    for (const [paramName, paramDef] of Object.entries(currentFlow.parameters)) {
                        if (paramDef.required && !this.flowParameters[paramName]) {
                            throw new Error(`Required parameter '${paramDef.label || paramName}' is missing`);
                        }
                    }
                }
            }

            const config = {
                mode: this.selectedFlow,
                primaryAgent: this.selectedAgent,
                selectedAgents: this.selectedAgents || [], // 包含选中的多个Agent
                parameters: this.flowParameters
            };

            // 最终调试日志
            this.logMessage(`Final config: ${JSON.stringify(config, null, 2)}`, 'info');

            // 尝试多个可能的API端点
            let response;
            let lastError;
            const endpoints = ['/api/flow-config', '/api/flow/configure', '/api/configure'];

            for (const endpoint of endpoints) {
                try {
                    this.logMessage(`Trying endpoint: ${endpoint}`, 'info');
                    response = await fetch(endpoint, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(config)
                    });

                    if (response.ok) {
                        this.logMessage(`✅ Connected successfully to ${endpoint}`, 'success');
                        break; // 成功找到有效端点
                    } else {
                        this.logMessage(`❌ ${endpoint} returned ${response.status}: ${response.statusText}`, 'warning');
                        lastError = `HTTP ${response.status}: ${response.statusText}`;
                    }
                } catch (error) {
                    this.logMessage(`❌ Failed to connect to ${endpoint}: ${error.message}`, 'warning');
                    lastError = error.message;
                    continue;
                }
            }

            if (!response || !response.ok) {
                // 如果所有API端点都失败，显示错误而不是演示模式
                this.logMessage('❌ All backend endpoints failed', 'error');
                this.logMessage(`Last error: ${lastError}`, 'error');
                this.updateStatus('error');

                // 询问用户是否要继续演示模式
                const userChoice = confirm(
                    '后端API不可用。是否要在演示模式下继续？\n' +
                    '在演示模式下，配置不会保存到后端，但您可以测试界面功能。'
                );

                if (userChoice) {
                    this.logMessage('⚠️ User chose to continue in demo mode', 'info');
                    this.updateStatus('configured');
                    this.logMessage('✅ Configuration applied successfully (Demo Mode)!', 'success');
                    this.logMessage(`Flow: ${this.getFlowName(this.selectedFlow)}`, 'success');
                    this.logMessage(`Agent: ${this.getAgentName(this.selectedAgent)}`, 'success');

                    // Log parameters if any
                    if (Object.keys(this.flowParameters).length > 0) {
                        this.logMessage('Parameters:', 'success');
                        Object.entries(this.flowParameters).forEach(([key, value]) => {
                            this.logMessage(`  ${key}: ${value}`, 'success');
                        });
                    }
                } else {
                    this.logMessage('❌ Configuration cancelled by user', 'error');
                    throw new Error('Backend API unavailable and user cancelled demo mode');
                }
                return;
            }

            const result = await response.json();

            if (result.success) {
                this.updateStatus('configured');
                this.logMessage('✅ Configuration applied successfully!', 'success');
                this.logMessage(`Flow: ${this.getFlowName(this.selectedFlow)}`, 'success');
                this.logMessage(`Agent: ${this.getAgentName(this.selectedAgent)}`, 'success');

                // Log parameters if any
                if (Object.keys(this.flowParameters).length > 0) {
                    this.logMessage('Parameters:', 'success');
                    Object.entries(this.flowParameters).forEach(([key, value]) => {
                        this.logMessage(`  ${key}: ${value}`, 'success');
                    });
                }
            } else {
                throw new Error(result.message || 'Configuration failed');
            }
        } catch (error) {
            this.updateStatus('error');
            this.logMessage(`❌ Configuration failed: ${error.message}`, 'error');
            console.error('Configuration error details:', error);
        }
    }

    resetConfiguration() {
        this.selectedFlow = 'single_agent';
        this.selectedAgent = 'Manus';
        this.flowParameters = {};

        document.getElementById('flow-selector').value = this.selectedFlow;
        document.getElementById('agent-selector').value = this.selectedAgent;

        this.updateFlowDescription();
        this.updateAgentDescription();
        this.updateParametersSection();
        this.updateStatus('not-configured');

        this.logMessage('🔄 Configuration reset', 'info');
    }

    getFlowName(flowId) {
        const flow = this.flows.find(f => f.id === flowId);
        return flow ? flow.name : flowId;
    }

    getAgentName(agentId) {
        const agent = this.agents.find(a => a.id === agentId);
        return agent ? agent.name : agentId;
    }

    logMessage(message, type = 'info') {
        // 尝试多个可能的日志容器
        const logContent = document.getElementById('log-content') ||
            document.getElementById('activityLog') ||
            document.getElementById('operation-logs');

        const timestamp = new Date().toLocaleTimeString();

        // 添加类型图标
        const typeIcons = {
            'info': 'ℹ️',
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'debug': '🔍'
        };

        const icon = typeIcons[type] || 'ℹ️';
        const formattedMessage = `${icon} ${message}`;

        if (logContent) {
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry log-${type}`;
            logEntry.innerHTML = `<span class="log-time">[${timestamp}]</span> ${formattedMessage}`;
            logContent.appendChild(logEntry);
            logContent.scrollTop = logContent.scrollHeight;
        } else {
            // 如果没有日志容器，使用console
            console.log(`[${type.toUpperCase()}] ${formattedMessage}`);
        }

        // 同时在浏览器控制台显示
        console.log(`[FlowAgentSelector-${type.toUpperCase()}] ${message}`);
    }

    updateStatus(status) {
        // 更新状态指示器
        const statusIndicator = document.getElementById('config-status') ||
            document.getElementById('configStatus');

        if (statusIndicator) {
            // 移除所有状态类
            statusIndicator.classList.remove('not-configured', 'configured', 'error');
            // 添加新状态类
            statusIndicator.classList.add(status);

            // 更新状态文本
            const statusText = statusIndicator.querySelector('.status-text') ||
                statusIndicator.querySelector('span');
            if (statusText) {
                const statusMessages = {
                    'not-configured': 'Not Configured',
                    'configured': 'Configured',
                    'error': 'Configuration Error'
                };
                statusText.textContent = statusMessages[status] || status;
            }
        }

        console.log(`Status updated to: ${status}`);
    }

    getFlowName(flowId) {
        const flow = this.availableFlows.find(f => f.name === flowId);
        return flow ? flow.label : flowId;
    }

    getAgentName(agentId) {
        const agent = this.agents.find(a => a.id === agentId);
        return agent ? agent.name : agentId;
    }

    // 测试连接方法
    async testConnection() {
        try {
            this.logMessage('Testing connection...', 'info');

            // 简单的连通性测试
            const response = await fetch('/api/health', {
                method: 'GET'
            });

            if (response.ok) {
                this.logMessage('✅ Connection test successful', 'success');
                this.updateStatus('configured');
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            this.logMessage(`❌ Connection test failed: ${error.message}`, 'error');
            this.updateStatus('error');
        }
    }

    // 更新多Agent选择区域的显示
    updateMultiAgentSection() {
        console.log('🤖 updateMultiAgentSection called');
        console.log('Selected flow:', this.selectedFlow);
        console.log('Available flows:', this.availableFlows);

        const multiAgentsSection = document.getElementById('multi-agents-section');
        const recommendedAgentsSection = document.getElementById('recommended-agents');

        console.log('Multi-agents section found:', !!multiAgentsSection);
        console.log('Recommended agents section found:', !!recommendedAgentsSection);

        if (!multiAgentsSection) {
            console.warn('Multi-agents section not found in DOM');
            return;
        }

        // 🎮 特殊处理Game Data Analysis - 强制显示多Agent选择
        if (this.selectedFlow === 'game_data_analysis') {
            console.log('🎮 Game Data Analysis: Force enabling multi-agent section');
            multiAgentsSection.style.display = 'block';

            // 清除所有checkbox的选中状态
            this.clearAllAgentSelections();

            // 自动选中Game Data Analysis的默认Agent
            const gameDataAgents = ['ExcelCleanAgent', 'MultiDataAnalysisCoordinator', 'KeyMetricAnalysisAgent', 'AnalysisResultQnAAgent'];
            console.log('Selecting Game Data Analysis default agents:', gameDataAgents);
            this.selectDefaultAgents(gameDataAgents);

            // 显示推荐Agent
            this.showRecommendedAgents(gameDataAgents);
            if (recommendedAgentsSection) {
                recommendedAgentsSection.style.display = 'block';
            }
            console.log('✅ Game Data Analysis multi-agent section configured');
            return;
        }

        // 📊 特殊处理Data Analysis Flow - 强制显示多Agent选择
        if (this.selectedFlow === 'data_analysis_flow') {
            console.log('📊 Data Analysis Flow: Force enabling multi-agent section');
            multiAgentsSection.style.display = 'block';

            // 清除所有checkbox的选中状态
            this.clearAllAgentSelections();

            // 自动选中Data Analysis的默认Agent
            const dataAnalysisAgents = ['DataAnalysisExpert', 'ExcelCleanAgent'];
            console.log('Selecting Data Analysis default agents:', dataAnalysisAgents);
            this.selectDefaultAgents(dataAnalysisAgents);

            // 显示推荐Agent (扩展推荐列表)
            const recommendedAgents = ['Manus', 'DataAnalysisExpert', 'ExcelCleanAgent', 'KeyMetricAnalysisAgent', 'AnalysisResultQnAAgent'];
            console.log('Showing Data Analysis recommended agents:', recommendedAgents);
            this.showRecommendedAgents(recommendedAgents);
            if (recommendedAgentsSection) {
                recommendedAgentsSection.style.display = 'block';
            }
            console.log('✅ Data Analysis Flow multi-agent section configured');
            return;
        }

        // 📋 通用逻辑：如果Flow支持多Agent，显示多Agent选择区域
        const currentFlow = this.availableFlows?.find(flow => flow.name === this.selectedFlow);
        console.log('Current flow for multi-agent:', currentFlow);

        if (currentFlow && currentFlow.supportMultipleAgents) {
            console.log('Flow supports multiple agents, showing section');
            multiAgentsSection.style.display = 'block';

            // 清除所有checkbox的选中状态
            this.clearAllAgentSelections();

            // 如果有默认Agent，自动选中
            if (currentFlow.defaultAgents && currentFlow.defaultAgents.length > 0) {
                console.log('Selecting default agents:', currentFlow.defaultAgents);
                this.selectDefaultAgents(currentFlow.defaultAgents);
            }

            // 显示推荐Agent
            if (currentFlow.recommendedAgents && currentFlow.recommendedAgents.length > 0) {
                console.log('Showing recommended agents:', currentFlow.recommendedAgents);
                this.showRecommendedAgents(currentFlow.recommendedAgents);
                if (recommendedAgentsSection) {
                    recommendedAgentsSection.style.display = 'block';
                }
            } else {
                if (recommendedAgentsSection) {
                    recommendedAgentsSection.style.display = 'none';
                }
            }
            console.log('✅ Multi-agent section configured for supportMultipleAgents flow');
        } else {
            console.log('Flow does not support multiple agents, hiding section');
            multiAgentsSection.style.display = 'none';
            if (recommendedAgentsSection) {
                recommendedAgentsSection.style.display = 'none';
            }
            this.selectedAgents = []; // 清空选中的Agent
        }
    }


    // 清除所有Agent checkbox的选中状态
    clearAllAgentSelections() {
        this.agents.forEach(agent => {
            const checkbox = document.getElementById(`agent-${agent.id}`);
            if (checkbox) {
                checkbox.checked = false;
            }
        });
        this.selectedAgents = [];
    }

    // 选中默认Agent
    selectDefaultAgents(defaultAgents) {
        defaultAgents.forEach(agentId => {
            const checkbox = document.getElementById(`agent-${agentId}`);
            if (checkbox) {
                checkbox.checked = true;
            }
        });
        this.updateSelectedAgents();
    }

    // 显示推荐Agent按钮
    showRecommendedAgents(recommendedAgents) {
        const buttonsContainer = document.getElementById('recommendation-buttons');
        if (!buttonsContainer) return;

        buttonsContainer.innerHTML = '';

        // 添加"选择推荐"按钮
        const selectAllBtn = document.createElement('button');
        selectAllBtn.className = 'btn btn-small btn-secondary';
        selectAllBtn.innerHTML = '✅ Select Recommended';
        selectAllBtn.onclick = () => this.selectRecommendedAgents(recommendedAgents);
        buttonsContainer.appendChild(selectAllBtn);

        // 添加"清除所有"按钮
        const clearAllBtn = document.createElement('button');
        clearAllBtn.className = 'btn btn-small btn-outline';
        clearAllBtn.innerHTML = '🗑️ Clear All';
        clearAllBtn.onclick = () => this.clearAllAgentSelections();
        buttonsContainer.appendChild(clearAllBtn);
    }

    // 选择推荐Agent
    selectRecommendedAgents(recommendedAgents) {
        this.clearAllAgentSelections();
        recommendedAgents.forEach(agentId => {
            const checkbox = document.getElementById(`agent-${agentId}`);
            if (checkbox) {
                checkbox.checked = true;
            }
        });
        this.updateSelectedAgents();
        this.logMessage(`已选择推荐的 ${recommendedAgents.length} 个Agent`, 'info');
    }

    // 更新选中的Agent列表
    updateSelectedAgents() {
        this.selectedAgents = [];
        this.agents.forEach(agent => {
            const checkbox = document.getElementById(`agent-${agent.id}`);
            if (checkbox && checkbox.checked) {
                this.selectedAgents.push(agent.id);
            }
        });

        console.log('Selected agents updated:', this.selectedAgents);
        this.logMessage(`已选择 ${this.selectedAgents.length} 个额外Agent: ${this.selectedAgents.join(', ')}`, 'info');
    }

    // 获取选中的Agent列表
    getSelectedAgents() {
        return this.selectedAgents || [];
    }
}

// 设置全局引用
window.FlowAgentSelector = FlowAgentSelector;
