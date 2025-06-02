
/**
 * ViralClip Pro v8.0 - Netflix-Level Team Collaboration Frontend
 * Real-time collaboration with permissions, comments, and workflows
 */

class NetflixLevelCollaborationHub {
    constructor() {
        this.websocket = null;
        this.currentWorkspace = 'main';
        this.currentProject = 'viral-video-1';
        this.currentUser = this.getCurrentUser();
        this.sessionId = null;
        this.collaborators = new Map();
        this.comments = [];
        this.versions = [];
        this.isConnected = false;
        
        this.init();
    }

    async init() {
        console.log('🚀 Initializing Netflix-level collaboration hub...');
        
        this.setupEventListeners();
        this.setupWebSocket();
        this.loadWorkspaceData();
        this.startHeartbeat();
        
        console.log('✅ Collaboration hub initialized');
    }

    setupEventListeners() {
        // Panel tab switching
        document.querySelectorAll('.panel-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchPanel(e.target.dataset.tab);
            });
        });

        // Project switching
        document.querySelectorAll('.project-item').forEach(item => {
            item.addEventListener('click', (e) => {
                this.switchProject(e.target.dataset.project);
            });
        });

        // Workspace switching
        document.querySelectorAll('.workspace-item').forEach(item => {
            item.addEventListener('click', (e) => {
                this.switchWorkspace(e.target.dataset.workspace);
            });
        });

        // Timeline interaction
        const timeline = document.querySelector('.timeline');
        if (timeline) {
            timeline.addEventListener('click', (e) => {
                this.handleTimelineClick(e);
            });
        }

        // Comment input
        const commentInput = document.querySelector('.comment-input');
        if (commentInput) {
            commentInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                    this.addComment();
                }
            });
        }

        // Window beforeunload
        window.addEventListener('beforeunload', () => {
            this.disconnect();
        });
    }

    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v8/collaboration/ws/${this.currentWorkspace}/${this.currentProject}`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('🔗 WebSocket connected');
            this.isConnected = true;
            this.updateConnectionStatus('🟢 Connected');
            this.joinCollaborationSession();
        };
        
        this.websocket.onmessage = (event) => {
            this.handleWebSocketMessage(JSON.parse(event.data));
        };
        
        this.websocket.onclose = () => {
            console.log('🔴 WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus('🔴 Disconnected');
            
            // Attempt reconnection
            setTimeout(() => {
                if (!this.isConnected) {
                    this.setupWebSocket();
                }
            }, 3000);
        };
        
        this.websocket.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
            this.showNotification('Connection error. Attempting to reconnect...', 'error');
        };
    }

    async joinCollaborationSession() {
        const message = {
            type: 'join_session',
            workspace_id: this.currentWorkspace,
            project_id: this.currentProject,
            user_id: this.currentUser.id,
            user_info: this.currentUser
        };
        
        this.sendWebSocketMessage(message);
    }

    handleWebSocketMessage(message) {
        console.log('📨 Received message:', message.type);
        
        switch (message.type) {
            case 'session_joined':
                this.sessionId = message.session_id;
                this.updateCollaborators(message.active_collaborators);
                break;
                
            case 'user_joined':
                this.addCollaborator(message);
                this.showNotification(`${message.username} joined the session`);
                break;
                
            case 'user_left':
                this.removeCollaborator(message.user_id);
                this.showNotification(`${message.username} left the session`);
                break;
                
            case 'real_time_operation':
                this.handleRealTimeOperation(message.operation);
                break;
                
            case 'comment_added':
                this.addCommentToUI(message.comment);
                break;
                
            case 'version_created':
                this.addVersionToUI(message.version);
                break;
                
            case 'mention_notification':
                this.handleMentionNotification(message);
                break;
                
            case 'review_request':
                this.handleReviewRequest(message);
                break;
                
            case 'project_state':
                this.loadProjectState(message.data);
                break;
                
            default:
                console.log('Unknown message type:', message.type);
        }
    }

    handleRealTimeOperation(operation) {
        console.log('⚡ Real-time operation:', operation.type);
        
        switch (operation.type) {
            case 'cursor_move':
                this.updateCursorPosition(operation.user_id, operation.position);
                break;
                
            case 'timeline_edit':
                this.updateTimeline(operation.changes);
                break;
                
            case 'content_edit':
                this.updateContent(operation.changes);
                break;
                
            case 'selection_change':
                this.updateSelection(operation.user_id, operation.selection);
                break;
        }
    }

    switchPanel(panelName) {
        // Update tab appearance
        document.querySelectorAll('.panel-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-tab="${panelName}"]`).classList.add('active');
        
        // Show/hide panels
        document.querySelectorAll('.panel-content > div').forEach(panel => {
            panel.style.display = 'none';
        });
        document.getElementById(`${panelName}Panel`).style.display = 'block';
    }

    switchProject(projectId) {
        if (projectId === this.currentProject) return;
        
        this.currentProject = projectId;
        
        // Update UI
        document.querySelectorAll('.project-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-project="${projectId}"]`).classList.add('active');
        
        // Reconnect WebSocket for new project
        if (this.websocket) {
            this.websocket.close();
        }
        this.setupWebSocket();
        
        this.loadProjectData(projectId);
    }

    switchWorkspace(workspaceId) {
        if (workspaceId === this.currentWorkspace) return;
        
        this.currentWorkspace = workspaceId;
        
        // Update UI
        document.querySelectorAll('.workspace-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-workspace="${workspaceId}"]`).classList.add('active');
        
        this.loadWorkspaceData();
    }

    async loadWorkspaceData() {
        try {
            const response = await fetch(`/api/v8/collaboration/workspace/${this.currentWorkspace}`);
            const data = await response.json();
            
            if (data.success) {
                this.updateWorkspaceUI(data.workspace);
            }
        } catch (error) {
            console.error('❌ Failed to load workspace data:', error);
            this.showNotification('Failed to load workspace data', 'error');
        }
    }

    async loadProjectData(projectId) {
        try {
            const response = await fetch(`/api/v8/collaboration/project/${this.currentWorkspace}/${projectId}`);
            const data = await response.json();
            
            if (data.success) {
                this.updateProjectUI(data.project);
            }
        } catch (error) {
            console.error('❌ Failed to load project data:', error);
        }
    }

    loadProjectState(state) {
        // Load comments
        this.comments = state.comments || [];
        this.updateCommentsUI();
        
        // Load version history
        if (state.latest_version) {
            this.addVersionToUI(state.latest_version);
        }
        
        // Load active collaborators
        this.updateCollaborators(state.active_collaborators || []);
    }

    updateCollaborators(collaborators) {
        this.collaborators.clear();
        collaborators.forEach(collab => {
            this.collaborators.set(collab.user_id, collab);
        });
        
        this.updateCollaboratorsUI();
        this.updateCursorsUI();
    }

    addCollaborator(collaborator) {
        this.collaborators.set(collaborator.user_id, collaborator);
        this.updateCollaboratorsUI();
    }

    removeCollaborator(userId) {
        this.collaborators.delete(userId);
        this.updateCollaboratorsUI();
        this.removeCursor(userId);
    }

    updateCollaboratorsUI() {
        const container = document.getElementById('collaboratorsList');
        if (!container) return;
        
        // Keep current user at top, add others
        const currentUserHtml = this.getCurrentUserHtml();
        const collaboratorsHtml = Array.from(this.collaborators.values())
            .filter(c => c.user_id !== this.currentUser.id)
            .map(c => this.getCollaboratorHtml(c))
            .join('');
        
        container.innerHTML = currentUserHtml + collaboratorsHtml;
    }

    getCurrentUserHtml() {
        return `
            <div class="collaborator-item">
                <div class="avatar">${this.getInitials(this.currentUser.username)}</div>
                <div class="collaborator-info">
                    <div class="collaborator-name">${this.currentUser.username}</div>
                    <div class="collaborator-status">Owner • Online</div>
                </div>
                <div class="online-indicator"></div>
            </div>
        `;
    }

    getCollaboratorHtml(collaborator) {
        const isOnline = this.collaborators.has(collaborator.user_id);
        const statusIndicator = isOnline ? '<div class="online-indicator"></div>' : '';
        const status = isOnline ? 'Online' : 'Away';
        
        return `
            <div class="collaborator-item">
                <div class="avatar">${this.getInitials(collaborator.username)}</div>
                <div class="collaborator-info">
                    <div class="collaborator-name">${collaborator.username}</div>
                    <div class="collaborator-status">Editor • ${status}</div>
                </div>
                ${statusIndicator}
            </div>
        `;
    }

    updateCursorPosition(userId, position) {
        const cursor = document.querySelector(`[data-user-id="${userId}"]`);
        if (cursor) {
            cursor.style.left = `${position * 100}%`;
        }
    }

    updateCursorsUI() {
        const container = document.querySelector('.collaboration-cursors');
        if (!container) return;
        
        const cursorsHtml = Array.from(this.collaborators.values())
            .filter(c => c.user_id !== this.currentUser.id && c.cursor_position)
            .map(c => this.getCursorHtml(c))
            .join('');
        
        container.innerHTML = cursorsHtml;
    }

    getCursorHtml(collaborator) {
        const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7'];
        const color = colors[collaborator.user_id.charCodeAt(0) % colors.length];
        
        return `
            <div class="cursor" 
                 style="left: ${(collaborator.cursor_position || 0) * 100}%; background: ${color};" 
                 data-user="${collaborator.username}"
                 data-user-id="${collaborator.user_id}">
            </div>
        `;
    }

    handleTimelineClick(event) {
        const timeline = event.currentTarget;
        const rect = timeline.getBoundingClientRect();
        const position = (event.clientX - rect.left) / rect.width;
        
        // Update own cursor
        const cursor = document.querySelector('.timeline-cursor');
        if (cursor) {
            cursor.style.left = `${position * 100}%`;
        }
        
        // Send cursor position to other collaborators
        this.sendRealTimeOperation({
            type: 'cursor_move',
            position: position
        });
    }

    sendRealTimeOperation(operation) {
        if (!this.isConnected || !this.sessionId) return;
        
        const message = {
            type: 'real_time_operation',
            session_id: this.sessionId,
            operation: operation
        };
        
        this.sendWebSocketMessage(message);
    }

    async addComment() {
        const input = document.querySelector('.comment-input');
        const content = input.value.trim();
        
        if (!content) return;
        
        // Extract mentions
        const mentions = this.extractMentions(content);
        
        try {
            const response = await fetch('/api/v8/collaboration/comment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    workspace_id: this.currentWorkspace,
                    project_id: this.currentProject,
                    user_id: this.currentUser.id,
                    content: content,
                    timestamp: Date.now() / 1000,
                    mentions: mentions
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                input.value = '';
                this.showNotification('Comment added successfully');
            }
        } catch (error) {
            console.error('❌ Failed to add comment:', error);
            this.showNotification('Failed to add comment', 'error');
        }
    }

    addCommentToUI(comment) {
        this.comments.push(comment);
        this.updateCommentsUI();
    }

    updateCommentsUI() {
        const container = document.getElementById('commentsList');
        if (!container) return;
        
        const commentsHtml = this.comments
            .sort((a, b) => b.timestamp - a.timestamp)
            .map(c => this.getCommentHtml(c))
            .join('');
        
        container.innerHTML = commentsHtml;
    }

    getCommentHtml(comment) {
        const time = this.formatTime(comment.timestamp);
        const content = this.formatCommentContent(comment.content);
        
        return `
            <div class="comment-item">
                <div class="comment-header">
                    <span class="comment-author">${comment.author_name}</span>
                    <span class="comment-time">${time}</span>
                </div>
                <div class="comment-content">${content}</div>
            </div>
        `;
    }

    async createVersion() {
        const message = prompt('Enter version message:');
        if (!message) return;
        
        try {
            const response = await fetch('/api/v8/collaboration/version', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    workspace_id: this.currentWorkspace,
                    project_id: this.currentProject,
                    user_id: this.currentUser.id,
                    message: message,
                    changes: this.getCurrentProjectState()
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Version created successfully');
            }
        } catch (error) {
            console.error('❌ Failed to create version:', error);
            this.showNotification('Failed to create version', 'error');
        }
    }

    addVersionToUI(version) {
        this.versions.push(version);
        this.updateVersionsUI();
    }

    updateVersionsUI() {
        const container = document.getElementById('versionList');
        if (!container) return;
        
        const versionsHtml = this.versions
            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
            .slice(0, 10) // Show last 10 versions
            .map(v => this.getVersionHtml(v))
            .join('');
        
        container.innerHTML = versionsHtml;
    }

    getVersionHtml(version) {
        const time = this.formatTime(new Date(version.timestamp).getTime() / 1000);
        
        return `
            <div class="version-item">
                <div class="version-meta">
                    <span class="version-author">${version.author_name}</span>
                    <span class="version-time">${time}</span>
                </div>
                <div>${version.message}</div>
            </div>
        `;
    }

    async shareProject() {
        try {
            const response = await fetch('/api/v8/collaboration/share', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    workspace_id: this.currentWorkspace,
                    project_id: this.currentProject,
                    user_id: this.currentUser.id,
                    expires_hours: 24,
                    permissions: ['view', 'comment']
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                navigator.clipboard.writeText(data.public_url);
                this.showNotification('Share link copied to clipboard!');
            }
        } catch (error) {
            console.error('❌ Failed to share project:', error);
            this.showNotification('Failed to create share link', 'error');
        }
    }

    async startApproval() {
        const reviewers = prompt('Enter reviewer usernames (comma-separated):');
        if (!reviewers) return;
        
        try {
            const response = await fetch('/api/v8/collaboration/approval', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    workspace_id: this.currentWorkspace,
                    project_id: this.currentProject,
                    created_by: this.currentUser.id,
                    reviewers: reviewers.split(',').map(r => r.trim()),
                    deadline_hours: 48
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Approval workflow started');
            }
        } catch (error) {
            console.error('❌ Failed to start approval:', error);
            this.showNotification('Failed to start approval workflow', 'error');
        }
    }

    async createWorkspace() {
        const name = prompt('Enter workspace name:');
        if (!name) return;
        
        try {
            const response = await fetch('/api/v8/collaboration/workspace', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name: name,
                    owner_id: this.currentUser.id,
                    description: ''
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Workspace created successfully');
                this.loadWorkspaceData();
            }
        } catch (error) {
            console.error('❌ Failed to create workspace:', error);
            this.showNotification('Failed to create workspace', 'error');
        }
    }

    async inviteUser() {
        const email = prompt('Enter user email:');
        if (!email) return;
        
        try {
            const response = await fetch('/api/v8/collaboration/invite', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    workspace_id: this.currentWorkspace,
                    email: email,
                    role: 'editor',
                    inviter_id: this.currentUser.id
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Invitation sent successfully');
            }
        } catch (error) {
            console.error('❌ Failed to invite user:', error);
            this.showNotification('Failed to send invitation', 'error');
        }
    }

    quickAction() {
        const actions = [
            { label: '💾 Save Version', action: () => this.createVersion() },
            { label: '💬 Add Comment', action: () => this.focusCommentInput() },
            { label: '🔗 Share Project', action: () => this.shareProject() },
            { label: '✅ Start Approval', action: () => this.startApproval() },
            { label: '👥 Invite User', action: () => this.inviteUser() }
        ];
        
        // Simple action menu (in production, use proper modal)
        const action = actions[Math.floor(Math.random() * actions.length)];
        action.action();
    }

    focusCommentInput() {
        this.switchPanel('comments');
        setTimeout(() => {
            const input = document.querySelector('.comment-input');
            if (input) input.focus();
        }, 100);
    }

    // Utility methods

    sendWebSocketMessage(message) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(message));
        }
    }

    disconnect() {
        if (this.websocket) {
            this.websocket.close();
        }
    }

    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            statusElement.textContent = status;
        }
    }

    showNotification(message, type = 'success') {
        const notification = document.getElementById('notification');
        const messageElement = document.getElementById('notificationMessage');
        
        if (notification && messageElement) {
            messageElement.textContent = message;
            notification.classList.add('show');
            
            setTimeout(() => {
                notification.classList.remove('show');
            }, 3000);
        }
    }

    getCurrentUser() {
        return {
            id: localStorage.getItem('userId') || 'user_' + Math.random().toString(36).substr(2, 9),
            username: localStorage.getItem('username') || 'Current User',
            email: localStorage.getItem('userEmail') || 'user@example.com',
            role: 'owner'
        };
    }

    getCurrentProjectState() {
        // Return current project state for versioning
        return {
            timeline_data: {},
            effects: [],
            transitions: [],
            audio_tracks: []
        };
    }

    extractMentions(content) {
        const mentions = [];
        const mentionRegex = /@(\w+)/g;
        let match;
        
        while ((match = mentionRegex.exec(content)) !== null) {
            mentions.push(match[1]);
        }
        
        return mentions;
    }

    formatCommentContent(content) {
        // Replace mentions with styled spans
        return content.replace(/@(\w+)/g, '<span style="color: #4ecdc4; font-weight: 600;">@$1</span>');
    }

    formatTime(timestamp) {
        const date = new Date(timestamp * 1000);
        const now = new Date();
        const diff = now - date;
        
        if (diff < 60000) return 'Just now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)} min ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)} hour ago`;
        return date.toLocaleDateString();
    }

    getInitials(name) {
        return name.split(' ').map(n => n[0]).join('').toUpperCase().substr(0, 2);
    }

    startHeartbeat() {
        setInterval(() => {
            if (this.isConnected) {
                this.sendWebSocketMessage({ type: 'heartbeat' });
            }
        }, 30000); // Every 30 seconds
    }

    handleMentionNotification(message) {
        this.showNotification(`You were mentioned in a comment by ${message.comment.author_name}`);
    }

    handleReviewRequest(message) {
        this.showNotification(`Review requested for project ${message.project_id}`);
    }
}

// Initialize collaboration hub
const collaborationHub = new NetflixLevelCollaborationHub();

// Export for global access
window.collaborationHub = collaborationHub;
