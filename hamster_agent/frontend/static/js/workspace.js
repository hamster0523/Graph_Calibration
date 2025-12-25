/**
 * 工作空间模块 - 文件管理和上传功能
 */

class WorkspaceManager {
    constructor() {
        this.fileList = document.getElementById('fileList');
        this.uploadBtn = document.getElementById('uploadFileBtn');
        this.fileInput = document.getElementById('fileInput');
        this.dropZone = document.getElementById('dropZone');
        this.refreshBtn = document.getElementById('refreshFilesBtn');
        this.createFolderBtn = document.getElementById('createFolderBtn');

        this.workspaceStats = {
            fileCount: document.getElementById('fileCount'),
            totalSize: document.getElementById('totalSize'),
            workspacePath: document.getElementById('workspacePath')
        };

        this.files = [];
        this.uploadQueue = [];
        this.isUploading = false;

        this.init();
        this.loadWorkspaceFiles();
    }

    init() {
        // 上传按钮
        this.uploadBtn?.addEventListener('click', () => {
            this.fileInput?.click();
        });

        // 文件选择
        this.fileInput?.addEventListener('change', (e) => {
            this.handleFileSelection(Array.from(e.target.files));
        });

        // 拖拽上传
        this.setupDragAndDrop();

        // 刷新按钮
        this.refreshBtn?.addEventListener('click', () => {
            this.loadWorkspaceFiles();
        });

        // 创建文件夹
        this.createFolderBtn?.addEventListener('click', () => {
            this.createFolder();
        });

        // 文件列表点击事件
        this.fileList?.addEventListener('click', (e) => {
            this.handleFileListClick(e);
        });
    }

    setupDragAndDrop() {
        if (!this.dropZone) return;

        // 阻止默认拖拽行为
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        // 拖拽视觉反馈
        ['dragenter', 'dragover'].forEach(eventName => {
            this.dropZone.addEventListener(eventName, () => {
                this.dropZone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.dropZone.addEventListener(eventName, () => {
                this.dropZone.classList.remove('dragover');
            });
        });

        // 文件拖拽放置
        this.dropZone.addEventListener('drop', (e) => {
            const files = Array.from(e.dataTransfer.files);
            this.handleFileSelection(files);
        });

        // 点击上传
        this.dropZone.addEventListener('click', () => {
            this.fileInput?.click();
        });
    }

    async loadWorkspaceFiles() {
        try {
            this.showLoading(true);
            const response = await window.apiClient.workspace.listFiles();
            this.files = response.files || [];
            this.updateWorkspaceStats(response);
            this.renderFileList();
        } catch (error) {
            console.error('Failed to load workspace files:', error);
            this.showError('Failed to load workspace files');
        } finally {
            this.showLoading(false);
        }
    }

    updateWorkspaceStats(data) {
        const totalSize = this.files.reduce((sum, file) => sum + file.size, 0);

        if (this.workspaceStats.fileCount) {
            this.workspaceStats.fileCount.textContent = this.files.length;
        }

        if (this.workspaceStats.totalSize) {
            this.workspaceStats.totalSize.textContent = this.formatFileSize(totalSize);
        }

        if (this.workspaceStats.workspacePath) {
            this.workspaceStats.workspacePath.textContent = data.workspace_path || 'Unknown';
        }
    }

    renderFileList() {
        if (!this.fileList) return;

        if (this.files.length === 0) {
            this.fileList.innerHTML = `
                <div class="empty-state">
                    <i class="icon-folder-open"></i>
                    <h3>No files in workspace</h3>
                    <p>Upload some files to get started</p>
                    <button class="btn-primary" onclick="document.getElementById('fileInput').click()">
                        <i class="icon-upload"></i>
                        Upload Files
                    </button>
                </div>
            `;
            return;
        }

        const fileElements = this.files.map(file => this.createFileElement(file));
        this.fileList.innerHTML = fileElements.join('');
    }

    createFileElement(file) {
        const iconClass = this.getFileIcon(file.name);
        const formattedSize = this.formatFileSize(file.size);
        const formattedDate = new Date(file.modified).toLocaleDateString();

        return `
            <div class="file-item" data-path="${file.path}">
                <div class="file-icon">
                    <i class="${iconClass}"></i>
                </div>
                <div class="file-info">
                    <div class="file-name">${this.escapeHtml(file.name)}</div>
                    <div class="file-meta">
                        <span class="file-size">${formattedSize}</span>
                        <span class="file-date">${formattedDate}</span>
                    </div>
                </div>
                <div class="file-actions">
                    <button class="action-btn download" data-action="download" title="Download">
                        <i class="icon-download"></i>
                    </button>
                    <button class="action-btn delete" data-action="delete" title="Delete">
                        <i class="icon-trash"></i>
                    </button>
                </div>
            </div>
        `;
    }

    handleFileListClick(e) {
        const action = e.target.closest('.action-btn')?.getAttribute('data-action');
        const fileItem = e.target.closest('.file-item');

        if (!action || !fileItem) return;

        const filePath = fileItem.getAttribute('data-path');
        const fileName = fileItem.querySelector('.file-name')?.textContent;

        switch (action) {
            case 'download':
                this.downloadFile(filePath, fileName);
                break;
            case 'delete':
                this.deleteFile(filePath, fileName);
                break;
        }
    }

    async handleFileSelection(files) {
        if (!files || files.length === 0) return;

        // 添加到上传队列
        this.uploadQueue.push(...files);

        if (!this.isUploading) {
            this.processUploadQueue();
        }
    }

    async processUploadQueue() {
        if (this.uploadQueue.length === 0) {
            this.isUploading = false;
            return;
        }

        this.isUploading = true;

        while (this.uploadQueue.length > 0) {
            const file = this.uploadQueue.shift();
            await this.uploadFile(file);
        }

        this.isUploading = false;
        await this.loadWorkspaceFiles(); // 刷新文件列表
    }

    async uploadFile(file) {
        let progressElement = null;

        try {
            console.log('Uploading file:', file.name);

            // 创建上传进度指示器
            progressElement = this.createUploadProgress(file.name);
            this.showUploadProgress(progressElement);

            const response = await window.apiClient.workspace.uploadFile(file, (progress) => {
                this.updateUploadProgress(progressElement, progress);
            });

            console.log('File uploaded successfully:', response);
            this.removeUploadProgress(progressElement);

            // 显示成功通知
            window.notificationManager?.show(`File "${file.name}" uploaded successfully`, 'success');

            // 刷新文件列表
            await this.loadWorkspaceFiles();

        } catch (error) {
            console.error('File upload failed:', error);

            // 安全地移除进度指示器
            if (progressElement) {
                this.removeUploadProgress(progressElement);
            }

            // 显示错误通知
            window.notificationManager?.show(`Failed to upload "${file.name}": ${error.message || error}`, 'error');
        }
    }

    async downloadFile(filePath, fileName) {
        try {
            const url = `/api/workspace/download/${encodeURIComponent(filePath)}`;

            // 创建下载链接
            const link = document.createElement('a');
            link.href = url;
            link.download = fileName;
            link.style.display = 'none';

            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            window.notificationManager?.show(`Downloading "${fileName}"`, 'info');

        } catch (error) {
            console.error('Download failed:', error);
            window.notificationManager?.show(`Failed to download "${fileName}"`, 'error');
        }
    }

    async deleteFile(filePath, fileName) {
        if (!confirm(`Are you sure you want to delete "${fileName}"?`)) return;

        try {
            await window.apiClient.workspace.deleteFile(filePath);
            await this.loadWorkspaceFiles(); // 刷新列表

            window.notificationManager?.show(`File "${fileName}" deleted successfully`, 'success');

        } catch (error) {
            console.error('Delete failed:', error);
            window.notificationManager?.show(`Failed to delete "${fileName}"`, 'error');
        }
    }

    async createFolder() {
        const folderName = prompt('Enter folder name:');
        if (!folderName?.trim()) return;

        try {
            // 创建文件夹的API调用（这里假设backend支持）
            // await window.apiClient.workspace.createFolder(folderName);

            // 暂时显示通知
            window.notificationManager?.show('Folder creation not implemented yet', 'info');

        } catch (error) {
            console.error('Create folder failed:', error);
            window.notificationManager?.show(`Failed to create folder "${folderName}"`, 'error');
        }
    }

    createUploadProgress(fileName) {
        const element = document.createElement('div');
        element.className = 'upload-progress';
        element.innerHTML = `
            <div class="upload-info">
                <span class="upload-name">${this.escapeHtml(fileName)}</span>
                <span class="upload-percent">0%</span>
            </div>
            <div class="upload-bar">
                <div class="upload-fill" style="width: 0%"></div>
            </div>
        `;
        return element;
    }

    showUploadProgress(element) {
        // 添加到工作空间顶部
        const workspaceHeader = document.querySelector('.workspace-header');
        if (workspaceHeader) {
            workspaceHeader.after(element);
        }
    }

    updateUploadProgress(element, progress) {
        const percent = Math.round(progress * 100);
        const percentElement = element.querySelector('.upload-percent');
        const fillElement = element.querySelector('.upload-fill');

        if (percentElement) percentElement.textContent = `${percent}%`;
        if (fillElement) fillElement.style.width = `${percent}%`;
    }

    removeUploadProgress(element) {
        setTimeout(() => {
            element?.remove();
        }, 2000);
    }

    showLoading(show) {
        if (!this.fileList) return;

        if (show) {
            this.fileList.innerHTML = `
                <div class="loading-state">
                    <div class="spinner"></div>
                    <p>Loading files...</p>
                </div>
            `;
        }
    }

    showError(message) {
        if (!this.fileList) return;

        this.fileList.innerHTML = `
            <div class="error-state">
                <i class="icon-alert-circle"></i>
                <h3>Error</h3>
                <p>${this.escapeHtml(message)}</p>
                <button class="btn-primary" onclick="window.workspaceManager.loadWorkspaceFiles()">
                    <i class="icon-refresh"></i>
                    Retry
                </button>
            </div>
        `;
    }

    getFileIcon(fileName) {
        const ext = fileName.split('.').pop()?.toLowerCase() || '';

        const iconMap = {
            // 代码文件
            'js': 'icon-file-code',
            'ts': 'icon-file-code',
            'py': 'icon-file-code',
            'java': 'icon-file-code',
            'cpp': 'icon-file-code',
            'html': 'icon-file-code',
            'css': 'icon-file-code',
            'json': 'icon-file-code',

            // 文档文件
            'txt': 'icon-file-text',
            'md': 'icon-file-text',
            'doc': 'icon-file-text',
            'docx': 'icon-file-text',
            'pdf': 'icon-file-text',

            // 图片文件
            'jpg': 'icon-file-image',
            'jpeg': 'icon-file-image',
            'png': 'icon-file-image',
            'gif': 'icon-file-image',
            'svg': 'icon-file-image',

            // 压缩文件
            'zip': 'icon-file-archive',
            'rar': 'icon-file-archive',
            '7z': 'icon-file-archive',
            'tar': 'icon-file-archive',

            // Excel文件
            'xls': 'icon-file-spreadsheet',
            'xlsx': 'icon-file-spreadsheet',
            'csv': 'icon-file-spreadsheet'
        };

        return iconMap[ext] || 'icon-file';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';

        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// 初始化工作空间管理器
document.addEventListener('DOMContentLoaded', () => {
    window.workspaceManager = new WorkspaceManager();
});

// 导出给其他模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WorkspaceManager;
}
