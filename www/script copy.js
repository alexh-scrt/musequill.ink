/**
 * MuseQuill.ink Book Planner Wizard
 * Interactive multi-step form for book creation parameters
 * Updated with structured sub-genre system
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
            throw error; // Don't continue without data
        }
    }

    /**
     * Populate all form fields with enum data
     */
    populateFormFields() {
        // Populate select fields
        this.populateSelect('genre', 'GenreType');
        this.populateSelect('sub_genre', 'SubGenre');
        this.populateSelect('length', 'BookLength');
        this.populateSelect('structure', 'StoryStructure');
        this.populateSelect('plot_type', 'PlotType');
        this.populateSelect('pov', 'NarrativePOV');
        this.populateSelect('pacing', 'PacingType');
        this.populateSelect('main_character_role', 'CharacterRole');
        this.populateSelect('character_archetype', 'CharacterArchetype');
        this.populateSelect('world_type', 'WorldType');
        this.populateSelect('magic_system', 'MagicSystemType');
        this.populateSelect('technology_level', 'TechnologyLevel');
        this.populateSelect('writing_style', 'WritingStyle');
        this.populateSelect('tone', 'ToneType');
        this.populateSelect('age_group', 'AgeGroup');
        this.populateSelect('audience_type', 'AudienceType');
        this.populateSelect('reading_level', 'ReadingLevel');
        this.populateSelect('publication_route', 'PublicationRoute');
        this.populateSelect('ai_assistance_level', 'AIAssistanceLevel');
        this.populateSelect('research_priority', 'ResearchPriority');
        this.populateSelect('writing_schedule', 'WritingSchedule');

        // Populate checkbox groups
        this.populateCheckboxGroup('conflict-types', 'ConflictType');
        this.populateCheckboxGroup('content-warnings', 'ContentWarning');
    }

    /**
     * Populate a select element with enum values
     */
    populateSelect(selectId, enumName) {
        const select = document.getElementById(selectId);
        if (!select) {
            console.warn(`Select element not found: ${selectId}`);
            return;
        }

        const enumValues = this.enumData[enumName] || [];

        enumValues.forEach(([value, label]) => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = label;
            select.appendChild(option);
        });
    }

    /**
     * Populate a checkbox group with enum values
     */
    populateCheckboxGroup(containerId, enumName) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.warn(`Container element not found: ${containerId}`);
            return;
        }

        const enumValues = this.enumData[enumName] || [];

        enumValues.forEach(([value, label]) => {
            const checkboxItem = document.createElement('div');
            checkboxItem.className = 'checkbox-item';
            checkboxItem.innerHTML = `
                <input type="checkbox" id="${value}" name="${containerId}" value="${value}">
                <label for="${value}">${label}</label>
            `;
            container.appendChild(checkboxItem);
        });
    }

    /**
     * Bind all event listeners
     */
    bindEvents() {
        // Navigation buttons
        document.getElementById('next-btn').addEventListener('click', () => this.nextStep());
        document.getElementById('prev-btn').addEventListener('click', () => this.prevStep());
        document.getElementById('submit-btn').addEventListener('click', () => this.submitForm());

        // Step indicator clicks
        document.querySelectorAll('.step').forEach(step => {
            step.addEventListener('click', (e) => {
                const stepNum = parseInt(e.target.dataset.step);
                if (stepNum <= this.currentStep || this.isStepCompleted(stepNum - 1)) {
                    this.goToStep(stepNum);
                }
            });
        });

        // Form field changes - UPDATED for structured sub-genre system
        document.getElementById('genre').addEventListener('change', async (e) => {
            await this.onGenreChange(e.target.value);
        });

        document.getElementById('sub_genre').addEventListener('change', async (e) => {
            await this.onSubGenreChange(e.target.value);
        });

        document.getElementById('world_type').addEventListener('change', (e) => {
            this.onWorldTypeChange(e.target.value);
        });

        document.getElementById('length').addEventListener('change', (e) => {
            this.onLengthChange(e.target.value);
        });

        // Real-time validation
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
     * Handle genre selection changes - UPDATED FOR STRUCTURED SUB-GENRES
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

        if (filteredSubGenres.length === 0) {
            this.showSubGenreMessage('No sub-genres available');
            console.warn(`❌ No fallback sub-genres found for: ${genre}`);
            return;
        }

        // Add filtered sub-genres
        filteredSubGenres.forEach(([value, label]) => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = label;
            subGenreSelect.appendChild(option);
        });

        console.log(`⚠️ Fallback: Added ${filteredSubGenres.length} sub-genres for ${genre}`);
    }

    /**
     * Get keywords for genre-based filtering
     */
    getGenreKeywords(genre) {
        const keywordMap = {
            'fantasy': ['fantasy', 'magical', 'epic', 'sword', 'dark', 'high', 'low', 'urban'],
            'science_fiction': ['science', 'space', 'cyber', 'steam', 'hard', 'soft', 'dystopian', 'time'],
            'mystery': ['mystery', 'detective', 'police', 'cozy', 'hard', 'locked'],
            'romance': ['romance', 'romantic', 'contemporary', 'historical', 'paranormal', 'erotic'],
            'thriller': ['thriller', 'psychological', 'legal', 'medical', 'espionage', 'techno'],
            'horror': ['horror', 'supernatural', 'psychological', 'gothic', 'cosmic', 'body'],
            'business': ['business', 'entrepreneurship', 'leadership', 'marketing', 'finance', 'management'],
            'self_help': ['personal', 'productivity', 'relationships', 'spirituality', 'career', 'parenting'],
            'technology': ['programming', 'software', 'data', 'artificial', 'cybersecurity', 'web'],
            'biography': ['political', 'celebrity', 'historical', 'business', 'sports', 'artist'],
            'memoir': ['family', 'travel', 'spiritual', 'addiction', 'illness', 'celebrity'],
            'history': ['ancient', 'medieval', 'modern', 'military', 'social', 'cultural'],
            'science': ['physics', 'biology', 'chemistry', 'astronomy', 'earth', 'environmental']
        };

        return keywordMap[genre] || [genre.replace('_', '')];
    }

    /**
     * Filter sub-genres by keywords
     */
    filterSubGenresByKeywords(allSubGenres, keywords) {
        return allSubGenres.filter(([value, label]) => {
            const valueLower = value.toLowerCase();
            const labelLower = label.toLowerCase();

            return keywords.some(keyword =>
                valueLower.includes(keyword.toLowerCase()) ||
                labelLower.includes(keyword.toLowerCase())
            );
        }).slice(0, 15); // Limit to 15 options
    }

    /**
     * Update world type visibility based on genre
     */
    updateWorldTypeVisibility(genre) {
        // This method can be expanded based on genre
        const magicSystemGroup = document.getElementById('magic-system-group');
        const fictionGenres = ['fantasy', 'science_fiction', 'horror', 'paranormal', 'urban_fantasy'];

        if (fictionGenres.includes(genre)) {
            // Enable magic system for fiction genres that might use it
            this.enableMagicSystemField();
        }
    }

    /**
     * Enable magic system field
     */
    enableMagicSystemField() {
        const magicSystemGroup = document.getElementById('magic-system-group');
        if (magicSystemGroup) {
            // This will be handled by the existing world type change handler
            // Just ensuring it's available
        }
    }

    /**
     * Validate sub-genre selection against genre
     */
    async validateSubGenreSelection(genre, subGenre) {
        if (!subGenre || subGenre === '') {
            return { valid: true, message: 'No sub-genre selected' };
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/validate/genre-subgenre`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ genre, subgenre: subGenre })
            });

            if (response.ok) {
                return await response.json();
            } else {
                return { valid: false, message: 'Validation service unavailable' };
            }
        } catch (error) {
            console.warn('Sub-genre validation failed:', error);
            return { valid: true, message: 'Validation skipped due to error' };
        }
    }

    /**
     * Handle world type selection changes
     */
    onWorldTypeChange(worldType) {
        const magicGroup = document.getElementById('magic-system-group');
        const fantasyWorlds = ['high_fantasy', 'low_fantasy', 'urban_fantasy'];

        if (fantasyWorlds.includes(worldType)) {
            magicGroup.style.display = 'block';
        } else {
            magicGroup.style.display = 'none';
            document.getElementById('magic_system').value = '';
        }
        this.validateCurrentStep();
    }

    /**
     * Handle length selection changes
     */
    onLengthChange(length) {
        this.updateMetadataDisplay('length', length);
        this.validateCurrentStep();
    }

    /**
     * Show genre-based recommendations
     */
    async showGenreRecommendations(genre) {
        const recommendationsBox = document.getElementById('recommendations-1');
        const tagsContainer = document.getElementById('genre-recommendations');

        try {
            // Get recommendations from backend
            const recommendations = await this.getGenreRecommendationsFromAPI(genre);

            if (recommendations && Object.keys(recommendations).length > 0) {
                recommendationsBox.style.display = 'block';

                // Create tags from recommendations object
                const tags = [];
                Object.entries(recommendations).forEach(([category, values]) => {
                    if (Array.isArray(values) && values.length > 0) {
                        // Take first few values from each category
                        values.slice(0, 2).forEach(value => {
                            tags.push(value.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()));
                        });
                    }
                });

                tagsContainer.innerHTML = tags.map(tag =>
                    `<span class="tag">${tag}</span>`
                ).join('');
            } else {
                recommendationsBox.style.display = 'none';
            }
        } catch (error) {
            console.warn('Failed to load recommendations:', error);
            recommendationsBox.style.display = 'none';
        }
    }

    /**
     * Get recommendations from the API
     */
    async getGenreRecommendationsFromAPI(genre) {
        try {
            // First try to get from already loaded data
            if (this.recommendations && this.recommendations[genre]) {
                return this.recommendations[genre];
            }

            // Otherwise fetch from API
            const response = await fetch(`${this.apiBaseUrl}/recommendations/${genre}`);
            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.warn('Failed to fetch recommendations from API:', error);
        }

        return {};
    }

    /**
     * Hide recommendations
     */
    hideRecommendations() {
        document.getElementById('recommendations-1').style.display = 'none';
    }

    /**
     * Update metadata display for form fields
     */
    updateMetadataDisplay(fieldType, value) {
        const metadataElement = document.getElementById(`${fieldType}-metadata`);
        if (!metadataElement) return;

        let metadata = '';

        if (fieldType === 'length') {
            metadata = this.getLengthInfoFromBackend(value);
        } else if (fieldType === 'genre') {
            metadata = this.getGenreMetadataFromBackend(value);
        }

        metadataElement.textContent = metadata;
    }

    /**
     * Clear metadata display for form fields
     */
    clearMetadataDisplay(fieldType) {
        const metadataElement = document.getElementById(`${fieldType}-metadata`);
        if (metadataElement) {
            metadataElement.textContent = '';
        }
    }

    /**
     * Get word count information from backend metadata
     */
    getLengthInfoFromBackend(length) {
        if (this.metadata && this.metadata.BookLength && this.metadata.BookLength.word_counts) {
            const wordCounts = this.metadata.BookLength.word_counts[length];
            if (wordCounts && Array.isArray(wordCounts) && wordCounts.length === 2) {
                return `Typically ${wordCounts[0].toLocaleString()}-${wordCounts[1].toLocaleString()} words`;
            }
        }

        // Fallback to enum data if metadata not available
        const enumData = this.enumData.BookLength || [];
        const lengthItem = enumData.find(([value]) => value === length);
        return lengthItem && lengthItem[1].includes('(') ?
            lengthItem[1].match(/\(([^)]+)\)/)?.[1] || '' : '';
    }

    /**
     * Get genre metadata from backend
     */
    getGenreMetadataFromBackend(genre) {
        if (this.metadata && this.metadata.GenreType && this.metadata.GenreType.popularity) {
            const popularity = this.metadata.GenreType.popularity[genre];
            if (popularity) {
                return `Popularity: ${popularity}/10 - ${popularity >= 8 ? 'Very popular' : popularity >= 6 ? 'Popular' : 'Niche market'}`;
            }
        }

        return 'Popular choice with good market demand';
    }

    /**
     * Handle checkbox group changes with limits
     */
    handleCheckboxGroupChange(group, checkbox) {
        const maxSelections = group.id === 'conflict-types' ? 3 : 999;
        const checkedBoxes = group.querySelectorAll('input[type="checkbox"]:checked');

        if (checkedBoxes.length > maxSelections) {
            checkbox.checked = false;
            this.showWarning(`You can select up to ${maxSelections} options.`);
        }

        this.validateCurrentStep();
    }

    /**
     * Show a temporary warning message
     */
    showWarning(message) {
        const warningContainer = document.getElementById('validation-warnings');
        warningContainer.innerHTML = `
            <div class="warning-box">
                <h4>⚠️ Warning</h4>
                <p>${message}</p>
            </div>
        `;

        setTimeout(() => {
            warningContainer.innerHTML = '';
        }, 3000);
    }

    /**
     * Show validation warning
     */
    showValidationWarning(field, message) {
        const warningContainer = document.getElementById('validation-warnings');
        if (!warningContainer) return;

        warningContainer.innerHTML = `
            <div class="warning-box">
                <h4>⚠️ Validation Issue</h4>
                <p><strong>${field}:</strong> ${message}</p>
            </div>
        `;

        setTimeout(() => {
            warningContainer.innerHTML = '';
        }, 5000);
    }

    /**
     * Show an error message
     */
    showError(message) {
        alert(`Error: ${message}`);
    }

    /**
     * Enhanced form validation that includes sub-genre validation
     */
    async validateCurrentStep() {
        const requiredFields = this.getRequiredFieldsForStep(this.currentStep);
        const basicValidation = requiredFields.every(fieldId => {
            const field = document.getElementById(fieldId);
            return field && field.value.trim() !== '';
        });

        let subGenreValidation = { valid: true };

        // Additional validation for step 1 (genre/sub-genre combination)
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
     * Update the visual display of steps
     */
    updateStepDisplay() {
        // Update step indicator
        document.querySelectorAll('.step').forEach((step, index) => {
            const stepNum = index + 1;
            step.classList.remove('active', 'completed');

            if (stepNum === this.currentStep) {
                step.classList.add('active');
            } else if (stepNum < this.currentStep) {
                step.classList.add('completed');
            }
        });

        // Update step panels
        document.querySelectorAll('.step-panel').forEach((panel, index) => {
            panel.classList.remove('active');
            if (index + 1 === this.currentStep) {
                panel.classList.add('active');
            }
        });

        // *** ADD THESE NEW LINES TO UPDATE THE NAVIGATION COUNTER ***

        // Update the step counter text "Step X of 6"
        const currentStepNumber = document.getElementById('current-step-number');
        if (currentStepNumber) {
            currentStepNumber.textContent = this.currentStep;
        }

        // Update the progress bar fill width
        const progressFill = document.getElementById('progress-fill');
        if (progressFill) {
            const progressPercentage = (this.currentStep / this.maxSteps) * 100;
            progressFill.style.width = `${progressPercentage}%`;

            // Also update the data-step attribute for CSS targeting if needed
            progressFill.setAttribute('data-step', this.currentStep);
        }

        // *** END OF NEW LINES ***

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
     * Debug method to test the new sub-genre system
     */
    async debugSubGenreSystem() {
        console.log('🧪 Testing Sub-Genre System');

        try {
            // Test the debug endpoint
            const response = await fetch(`${this.apiBaseUrl}/debug/subgenres`);
            const debugData = await response.json();

            console.log('📊 Debug data:', debugData);

            // Test specific genre
            const genreData = await fetch(`${this.apiBaseUrl}/subgenres/fantasy`);
            const fantasySubGenres = await genreData.json();

            console.log('🧙 Fantasy sub-genres:', fantasySubGenres);

            return {
                debugData,
                fantasySubGenres,
                status: 'success'
            };

        } catch (error) {
            console.error('❌ Debug test failed:', error);
            return {
                error: error.message,
                status: 'failed'
            };
        }
    }

    /**
     * Test sub-genre loading for a specific genre
     */
    async testSubGenreLoading(genre = 'fantasy') {
        console.log(`🧪 Testing sub-genre loading for: ${genre}`);

        try {
            const response = await fetch(`${this.apiBaseUrl}/subgenres/${genre}`);
            const data = await response.json();
            console.log('✅ Sub-genre data:', data);
            return data;
        } catch (error) {
            console.error('❌ Test failed:', error);
            return null;
        }
    }

    /**
     * Format enum value to readable label
     */
    formatEnumLabel(value) {
        return value
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
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