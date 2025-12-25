/**
 * 配置模块 - 管理系统配置和设置
 */

class ConfigManager {
    constructor() {
        this.configTabs = document.querySelectorAll('.tab-btn');
        this.configPanels = document.querySelectorAll('.config-panel');
        this.saveBtn = document.getElementById('saveConfigBtn');
        this.testBtn = document.getElementById('testConfigBtn');
        this.resetBtn = document.getElementById('resetConfigBtn');
        
        this.currentConfig = {};
        this.originalConfig = {};
        this.hasUnsavedChanges = false;
        
        this.init();
        this.loadConfiguration();
    }
    
    init() {
        // 标签页切换
        this.configTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                this.switchTab(tab.getAttribute('data-tab'));
            });
        });
        
        // 保存配置
        this.saveBtn?.addEventListener('click', () => {
            this.saveConfiguration();
        });
        
        // 测试配置
        this.testBtn?.addEventListener('click', () => {
            this.testConfiguration();
        });
        
        // 重置配置
        this.resetBtn?.addEventListener('click', () => {
            this.resetConfiguration();
        });
        
        // 监听配置变化
        this.setupConfigListeners();
        
        // 页面关闭前检查未保存变更
        window.addEventListener('beforeunload', (e) => {
            if (this.hasUnsavedChanges) {
                e.preventDefault();
                e.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
            }
        });
    }
    
    switchTab(tabName) {
        // 更新标签按钮状态
        this.configTabs.forEach(tab => {
            tab.classList.toggle('active', tab.getAttribute('data-tab') === tabName);
        });
        
        // 更新面板显示状态
        this.configPanels.forEach(panel => {
            panel.classList.toggle('active', panel.id === `${tabName}Panel`);
        });
    }
    
    setupConfigListeners() {
        // 监听表单元素变化
        const configContainer = document.querySelector('.config-content');
        if (!configContainer) return;
        
        configContainer.addEventListener('input', (e) => {
            if (e.target.matches('input, select, textarea')) {
                this.markAsChanged();
            }
        });
        
        configContainer.addEventListener('change', (e) => {
            if (e.target.matches('input[type="checkbox"], input[type="radio"]')) {
                this.markAsChanged();
            }
        });
        
        // 密码显示/隐藏
        document.getElementById('toggleApiKey')?.addEventListener('click', () => {
            this.togglePasswordVisibility('llmApiKey');
        });
    }
    
    async loadConfiguration() {
        try {
            this.showLoading(true);
            const config = await window.apiClient.config.getConfiguration();
            this.currentConfig = { ...config };
            this.originalConfig = { ...config };
            this.populateConfigForm(config);
            this.markAsUnchanged();
        } catch (error) {
            console.error('Failed to load configuration:', error);
            this.showError('Failed to load configuration');
        } finally {
            this.showLoading(false);
        }
    }
    
    populateConfigForm(config) {
        // LLM配置
        if (config.llm) {
            this.setFieldValue('llmModel', config.llm.model);
            this.setFieldValue('llmProvider', config.llm.provider);
            this.setFieldValue('llmBaseUrl', config.llm.base_url);
            this.setFieldValue('llmApiKey', config.llm.api_key);
            this.setFieldValue('llmMaxTokens', config.llm.max_tokens);
            this.setFieldValue('llmTemperature', config.llm.temperature);
        }
        
        // 浏览器配置
        if (config.browser) {
            this.setFieldValue('browserHeadless', config.browser.headless);
            this.setFieldValue('browserDisableSecurity', config.browser.disable_security);
        }
        
        // 搜索配置
        if (config.search) {
            this.setFieldValue('searchEngine', config.search.engine);
        }
        
        // 沙箱配置
        if (config.sandbox) {
            this.setFieldValue('sandboxEnabled', config.sandbox.enabled);
        }
        
        // 填充模型选项
        this.populateModelOptions(config.llm?.provider);
    }
    
    setFieldValue(fieldId, value) {
        const field = document.getElementById(fieldId);
        if (!field) return;
        
        if (field.type === 'checkbox') {
            field.checked = !!value;
        } else if (field.type === 'radio') {
            field.checked = field.value === value;
        } else {
            field.value = value || '';
        }
    }
    
    getFieldValue(fieldId) {
        const field = document.getElementById(fieldId);
        if (!field) return null;
        
        if (field.type === 'checkbox') {
            return field.checked;
        } else if (field.type === 'number') {
            return parseFloat(field.value) || 0;
        } else {
            return field.value;
        }
    }
    
    populateModelOptions(provider) {
        const modelSelect = document.getElementById('llmModel');
        if (!modelSelect) return;
        
        // 清空现有选项
        modelSelect.innerHTML = '<option value="">Select Model</option>';
        
        const models = {
            openai: [
                'gpt-4-turbo-preview',
                'gpt-4',
                'gpt-3.5-turbo',
                'gpt-3.5-turbo-16k'
            ],
            anthropic: [
                'claude-3-opus-20240229',
                'claude-3-sonnet-20240229',
                'claude-3-haiku-20240307',
                'claude-2.1',
                'claude-2.0'
            ],
            google: [
                'gemini-pro',
                'gemini-pro-vision'
            ]
        };
        
        const providerModels = models[provider] || [];
        providerModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelSelect.appendChild(option);
        });
    }
    
    async saveConfiguration() {
        try {
            this.showSaving(true);
            
            // 收集配置数据
            const config = {
                llm: {
                    model: this.getFieldValue('llmModel'),
                    provider: this.getFieldValue('llmProvider'),
                    base_url: this.getFieldValue('llmBaseUrl'),
                    api_key: this.getFieldValue('llmApiKey'),
                    max_tokens: this.getFieldValue('llmMaxTokens'),
                    temperature: this.getFieldValue('llmTemperature')
                },
                browser: {
                    headless: this.getFieldValue('browserHeadless'),
                    disable_security: this.getFieldValue('browserDisableSecurity')
                },
                search: {
                    engine: this.getFieldValue('searchEngine')
                },
                sandbox: {
                    enabled: this.getFieldValue('sandboxEnabled')
                }
            };
            
            // 验证配置
            const validation = this.validateConfiguration(config);
            if (!validation.valid) {
                this.showValidationErrors(validation.errors);
                return;
            }
            
            // 保存配置
            await window.apiClient.config.updateConfiguration(config);
            
            this.currentConfig = { ...config };
            this.originalConfig = { ...config };
            this.markAsUnchanged();
            
            window.notificationManager?.show('Configuration saved successfully', 'success');
            
        } catch (error) {
            console.error('Failed to save configuration:', error);
            window.notificationManager?.show('Failed to save configuration', 'error');
        } finally {
            this.showSaving(false);
        }
    }
    
    async testConfiguration() {
        try {
            this.showTesting(true);
            
            // 收集当前配置
            const config = {
                llm: {
                    model: this.getFieldValue('llmModel'),
                    provider: this.getFieldValue('llmProvider'),
                    base_url: this.getFieldValue('llmBaseUrl'),
                    api_key: this.getFieldValue('llmApiKey'),
                    max_tokens: this.getFieldValue('llmMaxTokens'),
                    temperature: this.getFieldValue('llmTemperature')
                }
            };
            
            const result = await window.apiClient.config.testConfiguration(config);
            
            if (result.valid) {
                window.notificationManager?.show('Configuration test passed', 'success');
            } else {
                window.notificationManager?.show(
                    `Configuration test failed: ${result.message}`, 
                    'error'
                );
            }
            
        } catch (error) {
            console.error('Configuration test failed:', error);
            window.notificationManager?.show('Configuration test failed', 'error');
        } finally {
            this.showTesting(false);
        }
    }
    
    resetConfiguration() {
        if (!confirm('Are you sure you want to reset all configuration to defaults?')) {
            return;
        }
        
        // 恢复到原始配置
        this.populateConfigForm(this.originalConfig);
        this.markAsUnchanged();
        
        window.notificationManager?.show('Configuration reset to last saved state', 'info');
    }
    
    validateConfiguration(config) {
        const errors = [];
        
        // 验证LLM配置
        if (config.llm) {
            if (!config.llm.provider) {
                errors.push('LLM provider is required');
            }
            if (!config.llm.model) {
                errors.push('LLM model is required');
            }
            if (!config.llm.api_key) {
                errors.push('API key is required');
            }
            if (config.llm.max_tokens <= 0) {
                errors.push('Max tokens must be greater than 0');
            }
            if (config.llm.temperature < 0 || config.llm.temperature > 2) {
                errors.push('Temperature must be between 0 and 2');
            }
        }
        
        return {
            valid: errors.length === 0,
            errors
        };
    }
    
    showValidationErrors(errors) {
        const errorList = errors.map(error => `<li>${error}</li>`).join('');
        window.notificationManager?.show(
            `<div>Configuration errors:<ul>${errorList}</ul></div>`,
            'error'
        );
    }
    
    togglePasswordVisibility(fieldId) {
        const field = document.getElementById(fieldId);
        const button = document.getElementById('toggleApiKey');
        
        if (!field || !button) return;
        
        if (field.type === 'password') {
            field.type = 'text';
            button.innerHTML = '<i class="icon-eye-off"></i>';
        } else {
            field.type = 'password';
            button.innerHTML = '<i class="icon-eye"></i>';
        }
    }
    
    markAsChanged() {
        this.hasUnsavedChanges = true;
        this.saveBtn?.classList.add('unsaved-changes');
        
        // 更新按钮文本
        if (this.saveBtn) {
            this.saveBtn.innerHTML = '<i class="icon-save"></i> Save Changes *';
        }
    }
    
    markAsUnchanged() {
        this.hasUnsavedChanges = false;
        this.saveBtn?.classList.remove('unsaved-changes');
        
        // 恢复按钮文本
        if (this.saveBtn) {
            this.saveBtn.innerHTML = '<i class="icon-save"></i> Save Changes';
        }
    }
    
    showLoading(show) {
        const configContent = document.querySelector('.config-content');
        if (!configContent) return;
        
        if (show) {
            configContent.style.opacity = '0.5';
            configContent.style.pointerEvents = 'none';
        } else {
            configContent.style.opacity = '1';
            configContent.style.pointerEvents = 'auto';
        }
    }
    
    showSaving(show) {
        if (this.saveBtn) {
            this.saveBtn.disabled = show;
            this.saveBtn.innerHTML = show ? 
                '<i class="icon-spinner spinning"></i> Saving...' :
                '<i class="icon-save"></i> Save Changes';
        }
    }
    
    showTesting(show) {
        if (this.testBtn) {
            this.testBtn.disabled = show;
            this.testBtn.innerHTML = show ? 
                '<i class="icon-spinner spinning"></i> Testing...' :
                '<i class="icon-test"></i> Test Config';
        }
    }
    
    showError(message) {
        window.notificationManager?.show(message, 'error');
    }
}

// 监听provider变化，更新模型选项
document.addEventListener('DOMContentLoaded', () => {
    const providerSelect = document.getElementById('llmProvider');
    if (providerSelect) {
        providerSelect.addEventListener('change', (e) => {
            const configManager = window.configManager;
            if (configManager) {
                configManager.populateModelOptions(e.target.value);
            }
        });
    }
    
    window.configManager = new ConfigManager();
});

// 导出给其他模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ConfigManager;
}