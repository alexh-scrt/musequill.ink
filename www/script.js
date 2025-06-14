/**
 * MuseQuill.ink Book Planner Wizard
 * Interactive multi-step form for book creation parameters
 * Enhanced with "Surprise Me" functionality for random book generation
 */

class BookPlannerWizard {
    constructor() {
        this.currentStep = 1;
        this.maxSteps = 6;
        this.formData = {};
        this.enumData = {};
        this.apiBaseUrl = this.getApiBaseUrl();

        this.init();
    }

    /**
     * Determine API base URL based on environment
     */
    getApiBaseUrl() {
        const hostname = window.location.hostname;
        const port = window.location.port;

        // Development environments
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            return `http://${hostname}:8055/api`; // Updated to match the API port
        }

        // Production - assume same domain
        return `${window.location.protocol}//${window.location.host}/api`;
    }

    /**
     * Initialize the wizard
     */
    async init() {
        try {
            await this.loadEnumData();
            this.populateFormFields();
            this.bindEvents();
            this.updateStepDisplay();
            console.log('MuseQuill Book Planner initialized successfully');
        } catch (error) {
            console.error('Failed to initialize wizard:', error);
            this.showError('Failed to load form data. Please refresh the page.');
        }
    }

    /**
     * Load enumeration data from API
     */
    async loadEnumData() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/enums`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.enumData = data.enums || data; // Handle different response formats
            this.metadata = data.metadata || {};
            this.recommendations = data.recommendations || {};

            console.log('Enum data loaded:', Object.keys(this.enumData).length, 'enums');
            console.log('Metadata loaded:', Object.keys(this.metadata).length, 'items');
            console.log('Recommendations loaded:', Object.keys(this.recommendations).length, 'genres');
        } catch (error) {
            console.error('Failed to load enum data from API:', error);
            this.showError('Failed to load form data from server. Please check your connection and refresh the page.');
            throw error;
        }
    }

    /**
     * Populate form fields with enum data
     */
    populateFormFields() {
        // Simple select fields
        this.populateSelect('genre', this.enumData.GenreType);
        this.populateSelect('length', this.enumData.BookLength);
        this.populateSelect('structure', this.enumData.StoryStructure);
        this.populateSelect('plot_type', this.enumData.PlotType);
        this.populateSelect('pov', this.enumData.NarrativePOV);
        this.populateSelect('pacing', this.enumData.PacingType);
        this.populateSelect('main_character_role', this.enumData.CharacterRole);
        this.populateSelect('character_archetype', this.enumData.CharacterArchetype);
        this.populateSelect('world_type', this.enumData.WorldType);
        this.populateSelect('magic_system', this.enumData.MagicSystemType);
        this.populateSelect('technology_level', this.enumData.TechnologyLevel);
        this.populateSelect('writing_style', this.enumData.WritingStyle);
        this.populateSelect('tone', this.enumData.ToneType);
        this.populateSelect('age_group', this.enumData.AgeGroup);
        this.populateSelect('audience_type', this.enumData.AudienceType);
        this.populateSelect('reading_level', this.enumData.ReadingLevel);
        this.populateSelect('publication_route', this.enumData.PublicationRoute);
        this.populateSelect('ai_assistance_level', this.enumData.AIAssistanceLevel);
        this.populateSelect('research_priority', this.enumData.ResearchPriority);
        this.populateSelect('writing_schedule', this.enumData.WritingSchedule);

        // Checkbox groups
        this.populateCheckboxGroup('conflict_types', this.enumData.ConflictType);
        this.populateCheckboxGroup('content_warnings', this.enumData.ContentWarning);

        console.log('Form fields populated successfully');
    }

    /**
     * Populate a select element with options
     */
    populateSelect(fieldId, options) {
        const select = document.getElementById(fieldId);
        if (!select || !options) return;

        // Clear existing options (except the first placeholder)
        while (select.children.length > 1) {
            select.removeChild(select.lastChild);
        }

        options.forEach(([value, label]) => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = label;
            select.appendChild(option);
        });
    }

    /**
     * Populate a checkbox group with options
     */
    populateCheckboxGroup(fieldId, options) {
        const container = document.getElementById(fieldId);
        if (!container || !options) return;

        container.innerHTML = '';

        options.forEach(([value, label]) => {
            const div = document.createElement('div');
            div.className = 'checkbox-item';

            const input = document.createElement('input');
            input.type = 'checkbox';
            input.id = `${fieldId}-${value}`;
            input.name = fieldId;
            input.value = value;

            const labelElement = document.createElement('label');
            labelElement.htmlFor = input.id;
            labelElement.textContent = label;

            div.appendChild(input);
            div.appendChild(labelElement);
            container.appendChild(div);
        });
    }

    /**
     * Bind event handlers
     */
    bindEvents() {
        // Navigation buttons
        document.getElementById('next-btn')?.addEventListener('click', () => this.nextStep());
        document.getElementById('prev-btn')?.addEventListener('click', () => this.prevStep());
        document.getElementById('submit-btn')?.addEventListener('click', () => this.submitForm());

        // Step indicator buttons
        document.querySelectorAll('.step[data-step]').forEach(button => {
            button.addEventListener('click', (e) => {
                const stepNum = parseInt(e.currentTarget.getAttribute('data-step'));
                this.goToStep(stepNum);
            });
        });

        // Genre change handler
        document.getElementById('genre')?.addEventListener('change', (e) => {
            this.onGenreChange(e.target.value);
        });

        // Sub-genre change handler
        document.getElementById('sub_genre')?.addEventListener('change', (e) => {
            this.onSubGenreChange(e.target.value);
        });

        // Form validation
        document.querySelectorAll('input, select, textarea').forEach(field => {
            field.addEventListener('change', () => this.validateCurrentStep());
            field.addEventListener('input', () => this.validateCurrentStep());
        });

        // Checkbox group handling
        document.querySelectorAll('.checkbox-group').forEach(group => {
            group.addEventListener('change', (e) => {
                if (e.target.type === 'checkbox') {
                    this.handleCheckboxGroupChange(group, e.target);
                }
            });
        });
    }

    /**
     * Handle genre selection changes
     */
    async onGenreChange(genre) {
        console.log(`🎭 Genre changed to: ${genre}`);

        if (genre) {
            // Show loading state for dependent fields
            this.setSubGenreLoading(true);

            try {
                // Update sub-genres using the new structured system
                await this.updateSubGenreOptions(genre);

                // Update other genre-dependent elements
                await this.showGenreRecommendations(genre);
                this.updateMetadataDisplay('genre', genre);
                this.updateWorldTypeVisibility(genre);

                console.log(`✅ Genre change handling complete for: ${genre}`);

            } catch (error) {
                console.error('❌ Error updating genre-related fields:', error);
                this.showError(`Failed to load options for ${genre}. Please try again.`);
            } finally {
                this.setSubGenreLoading(false);
            }
        } else {
            // Clear dependent fields when no genre selected
            this.clearSubGenreOptions();
            this.hideRecommendations();
            this.clearMetadataDisplay('genre');
        }

        this.validateCurrentStep();
    }

    /**
     * Handle sub-genre selection changes with validation
     */
    async onSubGenreChange(subGenre) {
        const genre = document.getElementById('genre')?.value;

        if (genre && subGenre) {
            try {
                const validation = await this.validateSubGenreSelection(genre, subGenre);
                if (!validation.valid) {
                    this.showValidationWarning('Sub-genre', validation.message || 'Invalid combination');
                    document.getElementById('sub_genre').value = ''; // Clear invalid selection
                }
            } catch (error) {
                console.warn('Sub-genre validation failed:', error);
            }
        }

        this.validateCurrentStep();
    }

    /**
     * Update sub-genre options using structured backend system
     */
    async updateSubGenreOptions(genre) {
        console.log(`🏷️  Updating sub-genres for genre: ${genre}`);

        const subGenreSelect = document.getElementById('sub_genre');
        if (!subGenreSelect) {
            console.warn('Sub-genre select element not found');
            return;
        }

        try {
            // Clear existing options first
            this.clearSubGenreOptions();

            // Fetch genre-specific sub-genres from the new API endpoint
            const response = await fetch(`${this.apiBaseUrl}/subgenres/${genre}`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log(`📊 Sub-genre API response:`, data);

            if (!data.success && data.count === 0) {
                console.warn(`No sub-genres available for genre: ${genre}`);
                this.showSubGenreMessage(`No specific sub-genres available for ${genre.replace('_', ' ')}`);
                return;
            }

            const subGenres = data.subgenres || [];

            if (subGenres.length === 0) {
                console.warn(`Empty sub-genres list for genre: ${genre}`);
                this.showSubGenreMessage('No sub-genres available');
                return;
            }

            // Populate the sub-genre dropdown
            subGenres.forEach(([value, label]) => {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = label;
                subGenreSelect.appendChild(option);
            });

            // Update metadata display
            this.updateSubGenreMetadata(data);

            console.log(`✅ Successfully loaded ${subGenres.length} sub-genres for ${genre}`);

        } catch (error) {
            console.error('❌ Failed to fetch sub-genres from API:', error);

            // Fallback: try to use cached enum data
            console.log('🔄 Attempting fallback to cached sub-genre data...');
            this.updateSubGenreOptionsFallback(genre);
        }
    }

    /**
     * Clear sub-genre options and reset to placeholder
     */
    clearSubGenreOptions() {
        const subGenreSelect = document.getElementById('sub_genre');
        if (!subGenreSelect) return;

        // Remove all options except the first one (placeholder)
        while (subGenreSelect.children.length > 1) {
            subGenreSelect.removeChild(subGenreSelect.lastChild);
        }

        // Reset selection to placeholder
        subGenreSelect.value = '';

        // Clear any metadata display
        this.clearSubGenreMetadata();
    }

    /**
     * Set loading state for sub-genre field
     */
    setSubGenreLoading(isLoading) {
        const subGenreSelect = document.getElementById('sub_genre');
        if (!subGenreSelect) return;

        if (isLoading) {
            subGenreSelect.disabled = true;

            // Add loading option
            const loadingOption = document.createElement('option');
            loadingOption.value = '';
            loadingOption.textContent = 'Loading sub-genres...';
            loadingOption.id = 'sub-genre-loading';
            loadingOption.disabled = true;
            subGenreSelect.appendChild(loadingOption);

        } else {
            subGenreSelect.disabled = false;

            // Remove loading option
            const loadingOption = document.getElementById('sub-genre-loading');
            if (loadingOption) {
                loadingOption.remove();
            }
        }
    }

    /**
     * Show a message in the sub-genre field
     */
    showSubGenreMessage(message) {
        const subGenreSelect = document.getElementById('sub_genre');
        if (!subGenreSelect) return;

        const messageOption = document.createElement('option');
        messageOption.value = '';
        messageOption.textContent = message;
        messageOption.disabled = true;
        messageOption.style.fontStyle = 'italic';
        messageOption.style.color = '#7f8c8d';
        subGenreSelect.appendChild(messageOption);
    }

    /**
     * Update sub-genre metadata display
     */
    updateSubGenreMetadata(data) {
        const metadataElement = document.getElementById('sub_genre-metadata');
        if (!metadataElement) return;

        if (data.count > 0) {
            metadataElement.textContent = `${data.count} specialized sub-genres available`;
            metadataElement.style.color = '#16a085';
        } else {
            metadataElement.textContent = 'General genre - no specific sub-genres';
            metadataElement.style.color = '#7f8c8d';
        }
    }

    /**
     * Clear sub-genre metadata display
     */
    clearSubGenreMetadata() {
        const metadataElement = document.getElementById('sub_genre-metadata');
        if (metadataElement) {
            metadataElement.textContent = '';
        }
    }

    /**
     * Fallback method using cached enum data
     */
    updateSubGenreOptionsFallback(genre) {
        console.log(`🔄 Using fallback sub-genre population for: ${genre}`);

        const subGenreSelect = document.getElementById('sub_genre');
        const allSubGenres = this.enumData.SubGenre || [];

        // Clear existing options
        this.clearSubGenreOptions();

        // Simple keyword-based filtering for fallback
        const genreKeywords = this.getGenreKeywords(genre);
        const filteredSubGenres = this.filterSubGenresByKeywords(allSubGenres, genreKeywords);

        if (filteredSubGenres.length > 0) {
            filteredSubGenres.forEach(([value, label]) => {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = label;
                subGenreSelect.appendChild(option);
            });

            console.log(`📝 Fallback loaded ${filteredSubGenres.length} sub-genres`);
        } else {
            this.showSubGenreMessage('No matching sub-genres found');
        }
    }

    /**
     * Get keywords for genre-based filtering
     */
    getGenreKeywords(genre) {
        const keywordMap = {
            fantasy: ['fantasy', 'magic', 'epic', 'urban', 'dark'],
            science_fiction: ['sci', 'space', 'cyber', 'steam', 'time'],
            mystery: ['mystery', 'detective', 'crime', 'cozy'],
            romance: ['romance', 'contemporary', 'historical', 'paranormal'],
            thriller: ['thriller', 'suspense', 'psychological'],
            horror: ['horror', 'supernatural', 'gothic'],
            business: ['business', 'entrepreneur', 'marketing', 'finance'],
            self_help: ['self', 'personal', 'productivity', 'wellness']
        };

        return keywordMap[genre] || [genre];
    }

    /**
     * Filter sub-genres by keywords
     */
    filterSubGenresByKeywords(subGenres, keywords) {
        return subGenres.filter(([value, label]) => {
            const text = `${value} ${label}`.toLowerCase();
            return keywords.some(keyword => text.includes(keyword.toLowerCase()));
        });
    }

    /**
     * Validate sub-genre selection
     */
    async validateSubGenreSelection(genre, subGenre) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/validate/genre-subgenre`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ genre, subgenre: subGenre })
            });

            if (response.ok) {
                return await response.json();
            } else {
                return { valid: false, message: 'Validation service unavailable' };
            }
        } catch (error) {
            console.warn('Sub-genre validation error:', error);
            return { valid: true }; // Assume valid if validation fails
        }
    }

    /**
     * Show genre recommendations
     */
    async showGenreRecommendations(genre) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/recommendations/${genre}`);
            if (response.ok) {
                const recommendations = await response.json();
                this.displayRecommendations(genre, recommendations);
            }
        } catch (error) {
            console.warn('Failed to load recommendations:', error);
        }
    }

    /**
     * Display recommendations panel
     */
    displayRecommendations(genre, recommendations) {
        const panel = document.getElementById('recommendations-panel');
        const genreSpan = document.getElementById('recommendations-genre');
        const content = document.getElementById('recommendations-content');

        if (!panel || !genreSpan || !content) return;

        genreSpan.textContent = this.formatEnumLabel(genre);

        let html = '';
        Object.entries(recommendations).forEach(([category, items]) => {
            if (items && items.length > 0) {
                html += `<div class="recommendation-category">
                    <h4>${this.formatEnumLabel(category)}</h4>
                    <ul>${items.map(item => `<li>${this.formatEnumLabel(item)}</li>`).join('')}</ul>
                </div>`;
            }
        });

        content.innerHTML = html;
        panel.classList.remove('hidden');
    }

    /**
     * Hide recommendations panel
     */
    hideRecommendations() {
        const panel = document.getElementById('recommendations-panel');
        if (panel) {
            panel.classList.add('hidden');
        }
    }

    /**
     * Update metadata display for a field
     */
    updateMetadataDisplay(fieldId, value) {
        const metadataElement = document.getElementById(`${fieldId}-metadata`);
        if (!metadataElement || !this.metadata) return;

        const fieldMetadata = this.metadata[fieldId];
        if (fieldMetadata && fieldMetadata[value]) {
            metadataElement.textContent = fieldMetadata[value];
            metadataElement.style.color = '#16a085';
        }
    }

    /**
     * Clear metadata display for a field
     */
    clearMetadataDisplay(fieldId) {
        const metadataElement = document.getElementById(`${fieldId}-metadata`);
        if (metadataElement) {
            metadataElement.textContent = '';
        }
    }

    /**
     * Update world type field visibility based on genre
     */
    updateWorldTypeVisibility(genre) {
        const magicSystemGroup = document.getElementById('magic_system_group');
        if (!magicSystemGroup) return;

        // Show magic system for fantasy-related genres
        const fantasyGenres = ['fantasy', 'paranormal', 'urban_fantasy'];
        const showMagicSystem = fantasyGenres.some(g => genre.includes(g));

        magicSystemGroup.style.display = showMagicSystem ? 'block' : 'none';
    }

    /**
     * Handle checkbox group changes with limits
     */
    handleCheckboxGroupChange(group, checkbox) {
        const maxSelections = group.dataset.maxSelections ? parseInt(group.dataset.maxSelections) : null;

        if (maxSelections) {
            const checked = group.querySelectorAll('input[type="checkbox"]:checked');

            if (checked.length > maxSelections) {
                checkbox.checked = false;
                this.showValidationWarning(
                    'Selection limit',
                    `You can select a maximum of ${maxSelections} options.`
                );
            }
        }

        this.validateCurrentStep();
    }

    /**
     * Show validation warning
     */
    showValidationWarning(field, message) {
        this.showError(`${field}: ${message}`);
    }

    /**
     * Show error message
     */
    showError(message) {
        const errorContainer = document.getElementById('error-container');
        const errorMessage = document.getElementById('error-message');

        if (errorContainer && errorMessage) {
            errorMessage.textContent = message;
            errorContainer.style.display = 'block';

            // Auto-hide after 5 seconds
            setTimeout(() => {
                errorContainer.style.display = 'none';
            }, 5000);
        }

        console.error('Wizard error:', message);
    }

    /**
     * Validate current step
     */
    async validateCurrentStep() {
        const requiredFields = this.getRequiredFieldsForStep(this.currentStep);
        let basicValidation = true;
        let subGenreValidation = { valid: true };

        // Check required fields
        for (const fieldId of requiredFields) {
            const field = document.getElementById(fieldId);
            if (!field || !field.value || field.value.trim() === '') {
                basicValidation = false;
                break;
            }
        }

        // Special validation for sub-genre if both genre and sub-genre are selected
        if (this.currentStep === 1) {
            const genre = document.getElementById('genre')?.value;
            const subGenre = document.getElementById('sub_genre')?.value;

            if (genre && subGenre) {
                subGenreValidation = await this.validateSubGenreSelection(genre, subGenre);

                if (!subGenreValidation.valid) {
                    this.showValidationWarning('Sub-genre selection', subGenreValidation.message);
                }
            }
        }

        const isValid = basicValidation && subGenreValidation.valid;

        // Update button states
        const nextBtn = document.getElementById('next-btn');
        const submitBtn = document.getElementById('submit-btn');

        if (this.currentStep === this.maxSteps) {
            submitBtn.disabled = !isValid;
        } else {
            nextBtn.disabled = !isValid;
        }

        return isValid;
    }

    /**
     * Get required fields for a specific step
     */
    getRequiredFieldsForStep(step) {
        const requiredFields = {
            1: ['title', 'genre', 'length'],
            2: ['structure', 'pov'],
            3: ['main_character_role', 'world_type'],
            4: ['writing_style', 'tone'],
            5: ['age_group', 'audience_type', 'publication_route'],
            6: ['ai_assistance_level']
        };
        return requiredFields[step] || [];
    }

    /**
     * Move to the next step
     */
    nextStep() {
        if (this.validateCurrentStep() && this.currentStep < this.maxSteps) {
            this.collectStepData();
            this.currentStep++;
            this.updateStepDisplay();
        }
    }

    /**
     * Move to the previous step
     */
    prevStep() {
        if (this.currentStep > 1) {
            this.currentStep--;
            this.updateStepDisplay();
        }
    }

    /**
     * Go to a specific step
     */
    goToStep(stepNum) {
        if (stepNum >= 1 && stepNum <= this.maxSteps) {
            this.collectStepData();
            this.currentStep = stepNum;
            this.updateStepDisplay();
        }
    }

    /**
     * Update step display and navigation
     */
    updateStepDisplay() {
        // Update step indicator
        document.querySelectorAll('.step').forEach((step, index) => {
            const stepNum = index + 1;
            step.classList.toggle('active', stepNum === this.currentStep);
            step.classList.toggle('completed', stepNum < this.currentStep);
            step.setAttribute('aria-selected', stepNum === this.currentStep);
        });

        // Update step panels
        document.querySelectorAll('.step-panel').forEach((panel, index) => {
            const stepNum = index + 1;
            panel.classList.toggle('active', stepNum === this.currentStep);
        });

        // Update step counter
        const stepCounter = document.getElementById('step-counter');
        if (stepCounter) {
            stepCounter.textContent = `Step ${this.currentStep} of ${this.maxSteps}`;
        }

        // Update navigation buttons
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');
        const submitBtn = document.getElementById('submit-btn');

        prevBtn.style.display = this.currentStep > 1 ? 'block' : 'none';
        nextBtn.style.display = this.currentStep < this.maxSteps ? 'block' : 'none';
        submitBtn.style.display = this.currentStep === this.maxSteps ? 'block' : 'none';

        // Validate current step
        this.validateCurrentStep();

        // Scroll to top of form
        document.querySelector('.step-content').scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }

    /**
     * Collect data from the current step
     */
    collectStepData() {
        const currentPanel = document.querySelector(`.step-panel[data-step="${this.currentStep}"]`);
        if (!currentPanel) return;

        const fields = currentPanel.querySelectorAll('input, select, textarea');

        fields.forEach(field => {
            if (field.type === 'checkbox') {
                if (!this.formData[field.name]) {
                    this.formData[field.name] = [];
                }
                // Clear and rebuild array to avoid duplicates
                this.formData[field.name] = Array.from(
                    currentPanel.querySelectorAll(`input[name="${field.name}"]:checked`)
                ).map(cb => cb.value);
            } else {
                this.formData[field.name] = field.value;
            }
        });
    }

    /**
     * Submit the complete form
     */
    async submitForm() {
        try {
            // Collect final step data
            this.collectStepData();

            // Validate all required fields
            if (!this.validateAllSteps()) {
                this.showError('Please complete all required fields before submitting.');
                return;
            }

            // Show loading state
            this.showLoadingState();

            // Submit to API
            const response = await fetch(`${this.apiBaseUrl}/books/create`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(this.formData)
            });

            if (response.ok) {
                const result = await response.json();
                this.showSuccess(result);
            } else {
                const errorData = await response.json().catch(() => ({ message: 'Unknown error' }));
                throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
            }
        } catch (error) {
            console.error('Submit error:', error);
            this.hideLoadingState();
            this.showError(`Failed to create book plan: ${error.message}. Please try again.`);
        }
    }

    /**
     * Validate all steps
     */
    validateAllSteps() {
        for (let step = 1; step <= this.maxSteps; step++) {
            const requiredFields = this.getRequiredFieldsForStep(step);
            const isValid = requiredFields.every(fieldId => {
                return this.formData[fieldId] && this.formData[fieldId] !== '';
            });

            if (!isValid) {
                this.goToStep(step);
                return false;
            }
        }
        return true;
    }

    /**
     * Show loading state
     */
    showLoadingState() {
        document.querySelector('.wizard').style.display = 'none';
        document.getElementById('loading').style.display = 'block';
    }

    /**
     * Hide loading state
     */
    hideLoadingState() {
        document.querySelector('.wizard').style.display = 'block';
        document.getElementById('loading').style.display = 'none';
    }

    /**
     * Show success message and redirect
     */
    showSuccess(result) {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('success').style.display = 'block';

        console.log('Book plan created successfully:', result);

        // Redirect after 3 seconds
        setTimeout(() => {
            const redirectUrl = result.book_id
                ? `/dashboard?book_id=${result.book_id}`
                : '/dashboard';
            window.location.href = redirectUrl;
        }, 3000);
    }

    /**
     * Check if a step is completed
     */
    isStepCompleted(stepNum) {
        const requiredFields = this.getRequiredFieldsForStep(stepNum);
        return requiredFields.every(fieldId => {
            return this.formData[fieldId] && this.formData[fieldId] !== '';
        });
    }

    /**
     * Get form data for debugging
     */
    getFormData() {
        this.collectStepData();
        return { ...this.formData };
    }

    /**
     * Reset the wizard
     */
    reset() {
        this.currentStep = 1;
        this.formData = {};

        // Clear all form fields
        document.querySelectorAll('input, select, textarea').forEach(field => {
            if (field.type === 'checkbox') {
                field.checked = false;
            } else {
                field.value = '';
            }
        });

        this.updateStepDisplay();
    }

    /**
     * Format enum value to readable label
     */
    formatEnumLabel(value) {
        return value
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    }

    // =============================================
    // SURPRISE ME FUNCTIONALITY
    // =============================================

    /**
     * Main surprise me function - orchestrates the entire random book creation process
     */
    async surpriseMe() {
        console.log('🎲 Starting Surprise Me book creation...');

        const surpriseBtn = document.getElementById('surprise-me-btn');
        if (!surpriseBtn) return;

        try {
            // Set loading state
            this.setSurpriseMeLoading(true);

            // Step 1: Load fresh enum data to ensure we have latest options
            console.log('📊 Loading enum data...');
            await this.loadEnumData();

            // Step 2: Generate random selections with smart combinations
            console.log('🎯 Generating smart random selections...');
            const randomSelections = await this.generateSmartRandomSelections();

            // Step 3: Validate the selections make sense together
            console.log('✅ Validating selections...');
            const validatedSelections = await this.validateAndRefineSelections(randomSelections);

            // Step 4: Populate all form fields
            console.log('📝 Populating form fields...');
            await this.populateFormWithSelections(validatedSelections);

            // Step 5: Submit the form directly (skip manual wizard steps)
            console.log('🚀 Submitting book creation request...');
            await this.submitSurpriseBook(validatedSelections);

        } catch (error) {
            console.error('❌ Surprise Me failed:', error);
            this.setSurpriseMeLoading(false);
            this.showError(`Surprise Me failed: ${error.message}`);
        }
    }

    /**
     * Generate smart random selections that work well together
     */
    async generateSmartRandomSelections() {
        const selections = {};

        try {
            // Debug: Log available enum data
            console.log('📊 Available enum data keys:', Object.keys(this.enumData));
            console.log('📊 WorldType data sample:', this.enumData.WorldType?.slice(0, 3));

            // Step 1: Book Basics - Start with genre since it affects other choices
            if (!this.enumData.GenreType) {
                throw new Error('GenreType enum data not available');
            }

            selections.genre = this.getRandomChoice(this.enumData.GenreType);
            if (!selections.genre) {
                throw new Error('Failed to select random genre');
            }
            console.log(`🎭 Selected genre: ${selections.genre}`);

            // Get subgenres for the selected genre
            try {
                const subgenreResponse = await fetch(`${this.apiBaseUrl}/subgenres/${selections.genre}`);
                const subgenreData = await subgenreResponse.json();

                if (subgenreData.subgenres && subgenreData.subgenres.length > 0) {
                    selections.sub_genre = this.getRandomChoice(subgenreData.subgenres);
                    console.log(`🏷️  Selected sub-genre: ${selections.sub_genre}`);
                }
            } catch (error) {
                console.warn('Could not load subgenres, continuing without sub-genre:', error.message);
            }

            // Generate a creative title based on genre
            selections.title = this.generateRandomTitle(selections.genre, selections.sub_genre);
            selections.subtitle = Math.random() > 0.6 ? this.generateRandomSubtitle(selections.genre) : '';

            // Safe enum selections with fallbacks
            selections.length = this.getRandomChoice(this.enumData.BookLength) || 'novel';
            selections.description = this.generateRandomDescription(selections.genre);

            // Step 2: Story Structure - Make choices that work with genre
            selections.structure = this.getRandomChoice(this.enumData.StoryStructure) || 'three_act';
            selections.plot_type = this.getRandomChoice(this.enumData.PlotType) || 'character_driven';
            selections.pov = this.getRandomChoice(this.enumData.NarrativePOV) || 'third_person_limited';
            selections.pacing = this.getRandomChoice(this.enumData.PacingType) || 'moderate';

            // Select 1-3 conflict types randomly
            const conflictCount = Math.floor(Math.random() * 3) + 1;
            selections.conflict_types = this.getRandomChoices(this.enumData.ConflictType, conflictCount) || [];

            // Step 3: Characters & World - Match with genre appropriately
            selections.main_character_role = this.getRandomChoice(this.enumData.CharacterRole) || 'protagonist';
            selections.character_archetype = this.getRandomChoice(this.enumData.CharacterArchetype) || 'hero';

            // This is where the error was occurring - now with better error handling
            try {
                selections.world_type = this.getGenreAppropriateWorldType(selections.genre);
                console.log(`🌍 Selected world type: ${selections.world_type}`);
            } catch (error) {
                console.error('Error selecting world type:', error);
                selections.world_type = 'realistic_contemporary'; // Safe fallback
            }

            // Magic system and technology based on genre and world type
            if (this.shouldHaveMagicSystem(selections.genre, selections.world_type)) {
                selections.magic_system = this.getRandomChoice(this.enumData.MagicSystemType);
            }

            if (this.shouldHaveTechnologyLevel(selections.genre, selections.world_type)) {
                selections.technology_level = this.getRandomChoice(this.enumData.TechnologyLevel);
            }

            // Step 4: Writing Style
            selections.writing_style = this.getRandomChoice(this.enumData.WritingStyle) || 'descriptive';
            selections.tone = this.getRandomChoice(this.enumData.ToneType) || 'neutral';

            // Step 5: Audience & Publishing - Make age group compatible choices
            selections.age_group = this.getRandomChoice(this.enumData.AgeGroup) || 'adult';
            console.log(`👥 Selected age group: ${selections.age_group}`);

            selections.audience_type = this.getAgeAppropriateAudienceType(selections.age_group);
            console.log(`🎯 Selected audience type: ${selections.audience_type}`);

            selections.reading_level = this.getRandomChoice(this.enumData.ReadingLevel) || 'intermediate';
            selections.publication_route = this.getRandomChoice(this.enumData.PublicationRoute) || 'self_publishing';

            // Content warnings - select 0-2 randomly, ensure age appropriate
            const warningCount = Math.floor(Math.random() * 3); // 0, 1, or 2
            selections.content_warnings = this.getAgeAppropriateContentWarnings(
                selections.age_group,
                warningCount
            );

            // Step 6: AI Assistance
            selections.ai_assistance_level = this.getRandomChoice(this.enumData.AIAssistanceLevel) || 'moderate';
            selections.research_priority = this.getRandomChoice(this.enumData.ResearchPriority) || 'plot_development';
            selections.writing_schedule = this.getRandomChoice(this.enumData.WritingSchedule) || 'flexible';
            selections.additional_notes = this.generateRandomAdditionalNotes(selections);

            console.log('✅ Smart random selections generated:', selections);
            return selections;

        } catch (error) {
            console.error('❌ Error in generateSmartRandomSelections:', error);
            throw new Error(`Failed to generate random selections: ${error.message}`);
        }
    }

    /**
     * Validate and refine random selections to ensure they make sense together
     */
    async validateAndRefineSelections(selections) {
        console.log('🔍 Validating random selections...');

        // Validate genre-subgenre combination
        if (selections.genre && selections.sub_genre) {
            try {
                const validation = await this.validateSubGenreSelection(
                    selections.genre,
                    selections.sub_genre
                );

                if (!validation.valid) {
                    console.warn('Invalid genre-subgenre combo, removing subgenre');
                    delete selections.sub_genre;
                }
            } catch (error) {
                console.warn('Could not validate genre-subgenre, removing subgenre');
                delete selections.sub_genre;
            }
        }

        // Validate all enum selections against backend values
        const enumValidations = {
            'genre': 'GenreType',
            'length': 'BookLength',
            'structure': 'StoryStructure',
            'plot_type': 'PlotType',
            'pov': 'NarrativePOV',
            'pacing': 'PacingType',
            'main_character_role': 'CharacterRole',
            'character_archetype': 'CharacterArchetype',
            'world_type': 'WorldType',
            'magic_system': 'MagicSystemType',
            'technology_level': 'TechnologyLevel',
            'writing_style': 'WritingStyle',
            'tone': 'ToneType',
            'age_group': 'AgeGroup',
            'audience_type': 'AudienceType',
            'reading_level': 'ReadingLevel',
            'publication_route': 'PublicationRoute',
            'ai_assistance_level': 'AIAssistanceLevel',
            'research_priority': 'ResearchPriority',
            'writing_schedule': 'WritingSchedule'
        };

        // Validate each enum field
        Object.entries(enumValidations).forEach(([fieldName, enumName]) => {
            if (selections[fieldName]) {
                const isValid = this.validateEnumValue(selections[fieldName], enumName);
                if (!isValid) {
                    console.warn(`Invalid ${fieldName} value: ${selections[fieldName]}, removing`);
                    delete selections[fieldName];
                }
            }
        });

        // Validate array fields (conflict_types, content_warnings)
        if (selections.conflict_types && Array.isArray(selections.conflict_types)) {
            selections.conflict_types = selections.conflict_types.filter(value =>
                this.validateEnumValue(value, 'ConflictType')
            );
        }

        if (selections.content_warnings && Array.isArray(selections.content_warnings)) {
            selections.content_warnings = selections.content_warnings.filter(value =>
                this.validateEnumValue(value, 'ContentWarning')
            );
        }

        // Refine content warnings based on age group
        if (selections.age_group === 'children' || selections.age_group === 'middle_grade') {
            const inappropriateWarnings = ['graphic_violence', 'sexual_content', 'substance_abuse'];
            selections.content_warnings = (selections.content_warnings || []).filter(warning =>
                !inappropriateWarnings.includes(warning)
            );
        }

        // Ensure magic system makes sense with world type
        if (selections.world_type === 'realistic_contemporary' ||
            selections.world_type === 'historical') {
            delete selections.magic_system;
        }

        // Ensure technology level makes sense
        if (selections.world_type === 'fantasy' || selections.world_type === 'medieval') {
            if (selections.technology_level === 'futuristic' ||
                selections.technology_level === 'advanced') {
                selections.technology_level = 'pre_industrial';
            }
        }

        console.log('✅ Validated selections:', selections);
        return selections;
    }

    /**
     * Validate that a value exists in the specified enum
     */
    validateEnumValue(value, enumName) {
        const enumData = this.enumData[enumName];
        if (!enumData || !Array.isArray(enumData)) {
            console.warn(`Enum ${enumName} not found or invalid`);
            return false;
        }

        return enumData.some(item =>
            Array.isArray(item) && item.length > 0 && item[0] === value
        );
    }

    /**
     * Populate all form fields with the random selections
     */
    async populateFormWithSelections(selections) {
        console.log('📝 Populating form fields with selections...');

        // Go to step 1 to start fresh
        this.currentStep = 1;
        this.updateStepDisplay();

        // Populate Step 1: Book Basics
        this.setFieldValue('title', selections.title);
        this.setFieldValue('subtitle', selections.subtitle || '');
        this.setFieldValue('genre', selections.genre);
        this.setFieldValue('length', selections.length);
        this.setFieldValue('description', selections.description || '');

        // Handle sub-genre after genre is set
        if (selections.sub_genre) {
            // Trigger genre change to load subgenres, then set subgenre
            await this.onGenreChange(selections.genre);
            // Wait a bit for the subgenres to load
            await new Promise(resolve => setTimeout(resolve, 500));
            this.setFieldValue('sub_genre', selections.sub_genre);
        }

        // Populate Step 2: Story Structure
        this.setFieldValue('structure', selections.structure);
        this.setFieldValue('plot_type', selections.plot_type);
        this.setFieldValue('pov', selections.pov);
        this.setFieldValue('pacing', selections.pacing);
        this.setCheckboxValues('conflict_types', selections.conflict_types);

        // Populate Step 3: Characters & World
        this.setFieldValue('main_character_role', selections.main_character_role);
        this.setFieldValue('character_archetype', selections.character_archetype);
        this.setFieldValue('world_type', selections.world_type);

        if (selections.magic_system) {
            this.setFieldValue('magic_system', selections.magic_system);
        }
        if (selections.technology_level) {
            this.setFieldValue('technology_level', selections.technology_level);
        }

        // Populate Step 4: Writing Style
        this.setFieldValue('writing_style', selections.writing_style);
        this.setFieldValue('tone', selections.tone);

        // Populate Step 5: Audience & Publishing
        this.setFieldValue('age_group', selections.age_group);
        console.log(`📝 Setting age_group field to: ${selections.age_group}`);

        this.setFieldValue('audience_type', selections.audience_type);
        console.log(`📝 Setting audience_type field to: ${selections.audience_type}`);

        this.setFieldValue('reading_level', selections.reading_level);
        this.setFieldValue('publication_route', selections.publication_route);
        this.setCheckboxValues('content_warnings', selections.content_warnings);

        // Populate Step 6: AI Assistance
        this.setFieldValue('ai_assistance_level', selections.ai_assistance_level);
        this.setFieldValue('research_priority', selections.research_priority);
        this.setFieldValue('writing_schedule', selections.writing_schedule);
        this.setFieldValue('additional_notes', selections.additional_notes || '');

        // Update form data object
        this.formData = { ...selections };

        console.log('✅ All form fields populated successfully');
    }

    /**
     * Submit the surprise book directly without going through wizard steps
     */
    async submitSurpriseBook(selections) {
        try {
            console.log('🚀 Submitting surprise book creation...');

            // Show loading state on the main wizard too
            this.showLoadingState();

            // Submit to API with all selections
            const response = await fetch(`${this.apiBaseUrl}/books/create`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(selections)
            });

            if (response.ok) {
                const result = await response.json();
                console.log('🎉 Surprise book created successfully!', result);

                // Show success and redirect
                this.setSurpriseMeLoading(false);
                this.showSuccess(result);

            } else {
                const errorData = await response.json().catch(() => ({ message: 'Unknown error' }));
                throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
            }

        } catch (error) {
            console.error('❌ Surprise book submission failed:', error);
            this.setSurpriseMeLoading(false);
            this.hideLoadingState();
            throw error;
        }
    }

    /**
     * Helper method to get a random choice from enum options
     */
    getRandomChoice(enumOptions) {
        if (!enumOptions || !Array.isArray(enumOptions) || enumOptions.length === 0) {
            console.warn('Invalid enum options provided to getRandomChoice:', enumOptions);
            return null;
        }

        // Filter out invalid entries
        const validOptions = enumOptions.filter(option =>
            Array.isArray(option) && option.length >= 1 && option[0] != null
        );

        if (validOptions.length === 0) {
            console.warn('No valid options found in enum data');
            return null;
        }

        const randomIndex = Math.floor(Math.random() * validOptions.length);
        return validOptions[randomIndex][0]; // Get the value, not the label
    }

    /**
     * Helper method to get multiple random choices from enum options
     */
    getRandomChoices(enumOptions, count) {
        if (!enumOptions || !Array.isArray(enumOptions) || enumOptions.length === 0) {
            console.warn('Invalid enum options provided to getRandomChoices:', enumOptions);
            return [];
        }

        // Filter out invalid entries
        const validOptions = enumOptions.filter(option =>
            Array.isArray(option) && option.length >= 1 && option[0] != null
        );

        if (validOptions.length === 0) {
            console.warn('No valid options found in enum data');
            return [];
        }

        const shuffled = [...validOptions].sort(() => 0.5 - Math.random());
        return shuffled.slice(0, Math.min(count, validOptions.length)).map(option => option[0]);
    }

    /**
     * Generate a creative random title based on genre
     */
    generateRandomTitle(genre, subgenre) {
        const titlePrefixes = {
            fantasy: ['The', 'Chronicles of', 'Tales of', 'The Legend of', 'Realm of'],
            science_fiction: ['Beyond', 'The', 'Voyage to', 'Wars of', 'The Last'],
            mystery: ['The', 'Murder at', 'The Case of', 'Death in', 'The'],
            romance: ['Love in', 'The', 'Hearts of', 'A', 'The Kiss of'],
            thriller: ['The', 'Blood', 'Dark', 'The Final', 'Shadow'],
            business: ['The', 'Mastering', 'The Art of', 'Building', 'The Future of'],
            self_help: ['The Power of', 'How to', 'The', 'Mastering', 'The Art of']
        };

        const titleNouns = {
            fantasy: ['Dragon', 'Kingdom', 'Crystal', 'Shadow', 'Magic', 'Sword', 'Crown'],
            science_fiction: ['Galaxy', 'Station', 'Planet', 'Star', 'Machine', 'Future', 'Colony'],
            mystery: ['Manor', 'Library', 'Express', 'Garden', 'Museum', 'Hotel', 'Study'],
            romance: ['Paris', 'Summer', 'Heart', 'Garden', 'Beach', 'Wedding', 'Dance'],
            thriller: ['Hunt', 'Game', 'Night', 'Secret', 'Truth', 'Conspiracy', 'Code'],
            business: ['Success', 'Innovation', 'Leadership', 'Growth', 'Strategy', 'Excellence'],
            self_help: ['Success', 'Happiness', 'Change', 'Life', 'Mind', 'Purpose', 'Growth']
        };

        const genreKey = Object.keys(titlePrefixes).find(key => genre.includes(key)) || 'fantasy';

        const prefix = this.getRandomChoice(titlePrefixes[genreKey].map(p => [p, p]));
        const noun = this.getRandomChoice(titleNouns[genreKey].map(n => [n, n]));

        return `${prefix} ${noun}`;
    }

    /**
     * Generate a random subtitle
     */
    generateRandomSubtitle(genre) {
        const subtitles = {
            fantasy: ['A Tale of Magic and Adventure', 'An Epic Fantasy', 'A Magical Journey'],
            science_fiction: ['A Space Opera', 'A Future Vision', 'A Sci-Fi Adventure'],
            mystery: ['A Detective Story', 'A Murder Mystery', 'A Puzzle Unsolved'],
            romance: ['A Love Story', 'A Romantic Adventure', 'A Tale of Hearts'],
            thriller: ['A Psychological Thriller', 'A Suspenseful Tale', 'A Race Against Time'],
            business: ['A Guide to Success', 'Strategies for Growth', 'A Business Blueprint'],
            self_help: ['Transform Your Life', 'A Guide to Success', 'Your Path to Growth']
        };

        const genreKey = Object.keys(subtitles).find(key => genre.includes(key)) || 'fantasy';
        return this.getRandomChoice(subtitles[genreKey].map(s => [s, s]));
    }

    /**
     * Generate a random description
     */
    generateRandomDescription(genre) {
        const descriptions = {
            fantasy: 'An epic tale of magic, adventure, and heroes rising to face impossible odds in a world where anything is possible.',
            science_fiction: 'A thrilling journey through space and time, exploring the possibilities of technology and the future of humanity.',
            mystery: 'A gripping puzzle that will keep readers guessing until the very last page, filled with clues, red herrings, and shocking revelations.',
            romance: 'A heartwarming story of love conquering all obstacles, with characters who will capture your heart and never let go.',
            thriller: 'A pulse-pounding adventure that will keep you on the edge of your seat from the first page to the last.',
            business: 'Essential strategies and insights for achieving success in today\'s competitive business environment.',
            self_help: 'A practical guide to personal transformation and achieving your full potential in life.'
        };

        const genreKey = Object.keys(descriptions).find(key => genre.includes(key)) || 'fantasy';
        return descriptions[genreKey];
    }

    /**
     * Get world type appropriate for genre
     */
    getGenreAppropriateWorldType(genre) {
        // Defensive check for enum data
        if (!this.enumData.WorldType || !Array.isArray(this.enumData.WorldType)) {
            console.warn('WorldType enum data not available, using fallback');
            return 'realistic_contemporary'; // Safe fallback
        }

        const worldTypeMapping = {
            fantasy: ['fantasy', 'medieval', 'alternate_history'],
            science_fiction: ['futuristic', 'space', 'dystopian'],
            historical: ['historical', 'alternate_history'],
            contemporary: ['realistic_contemporary', 'urban'],
            mystery: ['realistic_contemporary', 'historical'],
            romance: ['realistic_contemporary', 'historical', 'fantasy'],
            thriller: ['realistic_contemporary', 'futuristic'],
            business: ['realistic_contemporary'],
            self_help: ['realistic_contemporary']
        };

        const genreKey = Object.keys(worldTypeMapping).find(key =>
            genre && genre.includes && genre.includes(key)
        ) || 'contemporary';

        const appropriateWorlds = worldTypeMapping[genreKey] || ['realistic_contemporary'];

        // Find matching options from actual enum data with safety checks
        const availableWorlds = this.enumData.WorldType.filter(world => {
            // Safety check: ensure world is an array with at least one element
            if (!Array.isArray(world) || world.length === 0) {
                console.warn('Invalid world type data:', world);
                return false;
            }

            const worldValue = world[0];
            if (typeof worldValue !== 'string') {
                console.warn('Invalid world type value:', worldValue);
                return false;
            }

            return appropriateWorlds.includes(worldValue);
        });

        if (availableWorlds.length > 0) {
            return this.getRandomChoice(availableWorlds);
        } else {
            // Fallback: return a random world type from all available
            console.warn(`No appropriate worlds found for genre ${genre}, using random selection`);
            return this.getRandomChoice(this.enumData.WorldType);
        }
    }

    /**
     * Check if genre/world combo should have magic system
     */
    shouldHaveMagicSystem(genre, worldType) {
        const magicGenres = ['fantasy', 'paranormal'];
        const magicWorlds = ['fantasy', 'medieval', 'alternate_history'];

        return magicGenres.some(g => genre.includes(g)) ||
            magicWorlds.includes(worldType);
    }

    /**
     * Check if genre/world combo should have technology level
     */
    shouldHaveTechnologyLevel(genre, worldType) {
        const techGenres = ['science_fiction', 'cyberpunk', 'steampunk'];
        const techWorlds = ['futuristic', 'space', 'dystopian', 'realistic_contemporary'];

        return techGenres.some(g => genre.includes(g)) ||
            techWorlds.includes(worldType);
    }

    /**
     * Get age-appropriate audience type
     */
    getAgeAppropriateAudienceType(ageGroup) {
        // Debug logging
        console.log(`🎯 Getting audience type for age group: ${ageGroup}`);
        console.log(`📊 Available AudienceType data:`, this.enumData.AudienceType?.slice(0, 5));

        // Use actual enum values from the backend instead of hardcoded mappings
        if (!this.enumData.AudienceType || !Array.isArray(this.enumData.AudienceType)) {
            console.warn('AudienceType enum data not available, using fallback');
            return 'general_readers'; // Safe fallback that matches backend enum
        }

        // Get all available audience type values from the actual enum data
        const allAudienceTypes = this.enumData.AudienceType
            .filter(type => Array.isArray(type) && type.length > 0)
            .map(type => type[0]);

        console.log(`📋 All available audience types:`, allAudienceTypes);

        // Define age-appropriate mappings using actual backend enum values
        const ageMapping = {
            children: ['general_readers', 'parents', 'students'],
            middle_grade: ['general_readers', 'students', 'parents'],
            young_adult: ['general_readers', 'students', 'genre_fans'],
            adult: ['general_readers', 'genre_fans', 'professionals', 'academics', 'entrepreneurs', 'creatives', 'technical_audience']
        };

        const appropriateTypes = ageMapping[ageGroup] || ['general_readers'];
        console.log(`🎯 Age-appropriate types for ${ageGroup}:`, appropriateTypes);

        // Filter to only include types that actually exist in the backend enum
        const validAppropriateTypes = appropriateTypes.filter(type =>
            allAudienceTypes.includes(type)
        );

        console.log(`✅ Valid appropriate types:`, validAppropriateTypes);

        if (validAppropriateTypes.length > 0) {
            // Randomly select from the valid appropriate types
            const randomIndex = Math.floor(Math.random() * validAppropriateTypes.length);
            const selectedType = validAppropriateTypes[randomIndex];
            console.log(`🎲 Selected audience type: ${selectedType}`);
            return selectedType;
        } else {
            // Fallback: if none of our appropriate types exist, just pick 'general_readers' if it exists
            if (allAudienceTypes.includes('general_readers')) {
                console.warn(`No appropriate audience types found for age group ${ageGroup}, using general_readers`);
                return 'general_readers';
            } else {
                // Last resort: pick the first available audience type
                const fallback = allAudienceTypes[0] || 'general_readers';
                console.warn(`general_readers not found, using first available: ${fallback}`);
                return fallback;
            }
        }
    }

    /**
     * Get age-appropriate content warnings
     */
    getAgeAppropriateContentWarnings(ageGroup, count) {
        if (!this.enumData.ContentWarning || !Array.isArray(this.enumData.ContentWarning)) {
            console.warn('ContentWarning enum data not available');
            return [];
        }

        // Define age restrictions using actual backend enum values
        // We'll be more permissive and only exclude the most severe content for younger audiences
        const ageRestrictions = {
            children: ['graphic_violence', 'sexual_content', 'substance_abuse', 'strong_language'],
            middle_grade: ['graphic_violence', 'sexual_content', 'substance_abuse'],
            young_adult: ['graphic_violence'], // Only exclude the most severe
            adult: [] // No restrictions for adults
        };

        const forbidden = ageRestrictions[ageGroup] || [];

        // Filter out age-inappropriate warnings
        const availableWarnings = this.enumData.ContentWarning.filter(warning => {
            // Safety check for array structure
            if (!Array.isArray(warning) || warning.length === 0) {
                return false;
            }

            const warningValue = warning[0];
            return !forbidden.includes(warningValue);
        });

        if (availableWarnings.length === 0) {
            console.warn(`No appropriate content warnings found for age group ${ageGroup}`);
            return [];
        }

        return this.getRandomChoices(availableWarnings, count);
    }

    /**
     * Generate random additional notes
     */
    generateRandomAdditionalNotes(selections) {
        const notes = [
            'Looking forward to exploring unique character development in this story.',
            'Interested in incorporating current social themes into the narrative.',
            'Planning to focus on strong world-building and immersive descriptions.',
            'Want to create relatable characters that readers will connect with.',
            'Aiming for a fast-paced, engaging story that keeps readers hooked.',
            'Excited to blend traditional elements with fresh, modern perspectives.',
            'Planning to research extensively to ensure authenticity.',
            'Want to create a story that both entertains and inspires.',
            'Looking to challenge conventional genre expectations.',
            'Hoping to create memorable, quotable dialogue throughout.'
        ];

        return Math.random() > 0.5 ? this.getRandomChoice(notes.map(n => [n, n])) : '';
    }

    /**
     * Set form field value safely
     */
    setFieldValue(fieldName, value) {
        const field = document.getElementById(fieldName);
        if (field && value != null) {
            field.value = value;

            // Trigger change event to update any dependent fields
            field.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }

    /**
     * Set checkbox values for multi-select fields
     */
    setCheckboxValues(fieldName, values) {
        if (!values || !Array.isArray(values)) return;

        // Clear all checkboxes first
        const checkboxes = document.querySelectorAll(`input[name="${fieldName}"]`);
        checkboxes.forEach(cb => cb.checked = false);

        // Set selected values
        values.forEach(value => {
            const checkbox = document.querySelector(`input[name="${fieldName}"][value="${value}"]`);
            if (checkbox) {
                checkbox.checked = true;
            }
        });
    }

    /**
     * Set loading state for surprise me button
     */
    setSurpriseMeLoading(isLoading) {
        const surpriseBtn = document.getElementById('surprise-me-btn');
        if (!surpriseBtn) return;

        if (isLoading) {
            surpriseBtn.disabled = true;
            surpriseBtn.classList.add('loading');
            surpriseBtn.querySelector('.btn-text').textContent = '🎲 Creating...';
        } else {
            surpriseBtn.disabled = false;
            surpriseBtn.classList.remove('loading');
            surpriseBtn.querySelector('.btn-text').textContent = '🎲 Surprise Me!';
        }
    }
}

// Initialize the wizard when page loads
document.addEventListener('DOMContentLoaded', () => {
    try {
        window.bookPlannerWizard = new BookPlannerWizard();
    } catch (error) {
        console.error('Failed to initialize BookPlannerWizard:', error);
        alert('Failed to initialize the book planner. Please refresh the page.');
    }
});

// Expose wizard for debugging - ENHANCED WITH SUB-GENRE TESTING
if (typeof window !== 'undefined') {
    window.MuseQuillDebug = {
        getWizard: () => window.bookPlannerWizard,
        getFormData: () => window.bookPlannerWizard?.getFormData(),
        resetWizard: () => window.bookPlannerWizard?.reset(),
        testSurpriseMe: () => window.bookPlannerWizard?.surpriseMe(),

        // Enhanced debugging methods
        inspectEnumData: () => {
            const wizard = window.bookPlannerWizard;
            if (!wizard) return { error: 'Wizard not initialized' };

            const enumData = wizard.enumData;
            const inspection = {};

            Object.keys(enumData).forEach(key => {
                const data = enumData[key];
                inspection[key] = {
                    type: Array.isArray(data) ? 'array' : typeof data,
                    length: Array.isArray(data) ? data.length : 'N/A',
                    sample: Array.isArray(data) ? data.slice(0, 3) : data,
                    structure: Array.isArray(data) && data.length > 0 ?
                        `[${typeof data[0]}${Array.isArray(data[0]) ? `[${data[0].length}]` : ''}]` : 'unknown',
                    allValues: Array.isArray(data) ? data.map(item => Array.isArray(item) ? item[0] : item) : 'N/A'
                };
            });

            return inspection;
        },

        // Debug audience type selection specifically with live testing
        debugAudienceType: () => {
            const wizard = window.bookPlannerWizard;
            if (!wizard) return { error: 'Wizard not initialized' };

            console.log('🧪 Testing audience type selection...');

            const ageGroups = ['children', 'middle_grade', 'young_adult', 'adult'];
            const results = {};

            // Test the enum data first
            console.log('📊 Raw AudienceType enum data:', wizard.enumData.AudienceType);

            const allAudienceTypes = wizard.enumData.AudienceType
                ?.filter(type => Array.isArray(type) && type.length > 0)
                ?.map(type => type[0]) || [];

            console.log('📋 Extracted audience type values:', allAudienceTypes);

            ageGroups.forEach(ageGroup => {
                try {
                    console.log(`\n🎯 Testing age group: ${ageGroup}`);
                    const audienceType = wizard.getAgeAppropriateAudienceType(ageGroup);
                    const isValid = wizard.validateEnumValue(audienceType, 'AudienceType');

                    results[ageGroup] = {
                        selected: audienceType,
                        isValid: isValid
                    };

                    console.log(`✅ Result for ${ageGroup}: ${audienceType} (valid: ${isValid})`);
                } catch (error) {
                    results[ageGroup] = { error: error.message };
                    console.error(`❌ Error for ${ageGroup}: ${error.message}`);
                }
            });

            return {
                results,
                availableAudienceTypes: allAudienceTypes,
                enumDataStructure: wizard.enumData.AudienceType?.slice(0, 3)
            };
        },

        // Quick test to see what audience type gets selected for a specific age
        quickAudienceTest: (ageGroup = 'middle_grade') => {
            const wizard = window.bookPlannerWizard;
            if (!wizard) return { error: 'Wizard not initialized' };

            console.log(`🎯 Quick test for age group: ${ageGroup}`);
            const result = wizard.getAgeAppropriateAudienceType(ageGroup);
            console.log(`🎲 Selected: ${result}`);

            return {
                ageGroup,
                selected: result,
                isValid: wizard.validateEnumValue(result, 'AudienceType')
            };
        },

        // Test surprise me with detailed logging
        testSurpriseWithLogging: async () => {
            const wizard = window.bookPlannerWizard;
            if (!wizard) return { error: 'Wizard not initialized' };

            try {
                console.log('🧪 Starting test surprise generation...');
                const selections = await wizard.generateSmartRandomSelections();

                console.log('🔍 Pre-validation selections:', selections);
                const validatedSelections = await wizard.validateAndRefineSelections(selections);

                console.log('✅ Post-validation selections:', validatedSelections);

                // Validate each field against backend enums
                const validationResults = {};
                const enumMappings = {
                    'audience_type': 'AudienceType',
                    'age_group': 'AgeGroup',
                    'genre': 'GenreType',
                    'world_type': 'WorldType'
                };

                Object.entries(enumMappings).forEach(([field, enumName]) => {
                    if (validatedSelections[field]) {
                        validationResults[field] = {
                            value: validatedSelections[field],
                            isValid: wizard.validateEnumValue(validatedSelections[field], enumName),
                            availableValues: wizard.enumData[enumName]?.map(item => item[0]) || []
                        };
                    }
                });

                return {
                    selections: validatedSelections,
                    validationResults,
                    status: 'success'
                };

            } catch (error) {
                return { error: error.message, status: 'failed' };
            }
        },

        // Test enum selection safely
        testRandomSelection: (enumKey) => {
            const wizard = window.bookPlannerWizard;
            if (!wizard) return { error: 'Wizard not initialized' };

            const enumData = wizard.enumData[enumKey];
            if (!enumData) return { error: `Enum ${enumKey} not found` };

            try {
                const selection = wizard.getRandomChoice(enumData);
                return {
                    enumKey,
                    selection,
                    enumLength: enumData.length,
                    enumSample: enumData.slice(0, 3)
                };
            } catch (error) {
                return {
                    error: error.message,
                    enumKey,
                    enumData: enumData.slice(0, 3)
                };
            }
        },

        // New sub-genre testing methods
        testSubGenres: () => window.bookPlannerWizard?.debugSubGenreSystem(),
        getSubGenresFor: (genre) => fetch(`${window.bookPlannerWizard?.apiBaseUrl}/subgenres/${genre}`).then(r => r.json()),
        validateGenreSubGenre: (genre, subgenre) =>
            fetch(`${window.bookPlannerWizard?.apiBaseUrl}/validate/genre-subgenre`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ genre, subgenre })
            }).then(r => r.json()),

        // Test specific genre sub-genre loading
        testGenre: (genre) => window.bookPlannerWizard?.testSubGenreLoading(genre),

        // Test API connectivity
        testAPI: async () => {
            const wizard = window.bookPlannerWizard;
            if (!wizard) return { error: 'Wizard not initialized' };

            try {
                const health = await fetch(`${wizard.apiBaseUrl}/health`).then(r => r.json());
                const enums = await fetch(`${wizard.apiBaseUrl}/enums`).then(r => r.json());
                const fantasy = await fetch(`${wizard.apiBaseUrl}/subgenres/fantasy`).then(r => r.json());

                return {
                    health,
                    enumCount: Object.keys(enums.enums || {}).length,
                    fantasySubGenres: fantasy.count || 0,
                    status: 'success'
                };
            } catch (error) {
                return { error: error.message, status: 'failed' };
            }
        }
    };
}