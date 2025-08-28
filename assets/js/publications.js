// Publications page JavaScript functionality

(function() {
    'use strict';
    
    // DOM elements
    let publicationItems = [];
    let searchInput = null;
    let typeFilter = null;
    let yearFilter = null;
    let sortFilter = null;
    let clearFiltersBtn = null;
    let resultsCount = null;
    
    // Search and filter state
    let currentFilters = {
        search: '',
        type: 'all',
        year: 'all',
        sort: 'date-desc'
    };
    
    // Initialize when DOM is loaded
    document.addEventListener('DOMContentLoaded', function() {
        initializeElements();
        initializeSearch();
        initializeFilters();
        initializeSorting();
        
        // Initial filter application
        applyFilters();
        
        // Initialize other interactive features
        initializeAnimations();
        initializeIntersectionObserver();
    });
    
    function initializeElements() {
        // Get all publication items
        publicationItems = Array.from(document.querySelectorAll('.publication-item'));
        
        // Get filter elements
        searchInput = document.getElementById('search-input');
        typeFilter = document.getElementById('type-filter');
        yearFilter = document.getElementById('year-filter');
        sortFilter = document.getElementById('sort-filter');
        clearFiltersBtn = document.getElementById('clear-filters');
        
        // Create results count element if it doesn't exist
        if (!document.querySelector('.results-count')) {
            resultsCount = document.createElement('div');
            resultsCount.className = 'results-count';
            const filterNav = document.querySelector('.filter-nav .container');
            if (filterNav) {
                filterNav.appendChild(resultsCount);
            }
        } else {
            resultsCount = document.querySelector('.results-count');
        }
    }
    
    function initializeSearch() {
        if (!searchInput) return;
        
        // Real-time search with debouncing
        let searchTimeout;
        searchInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                currentFilters.search = this.value.toLowerCase().trim();
                applyFilters();
            }, 300);
        });
        
        // Search on Enter key
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                clearTimeout(searchTimeout);
                currentFilters.search = this.value.toLowerCase().trim();
                applyFilters();
            }
        });
    }
    
    function initializeFilters() {
        // Type filter
        if (typeFilter) {
            typeFilter.addEventListener('change', function() {
                currentFilters.type = this.value;
                applyFilters();
            });
        }
        
        // Year filter
        if (yearFilter) {
            yearFilter.addEventListener('change', function() {
                currentFilters.year = this.value;
                applyFilters();
            });
        }
        
        // Clear filters button
        if (clearFiltersBtn) {
            clearFiltersBtn.addEventListener('click', function() {
                clearAllFilters();
            });
        }
    }
    
    function initializeSorting() {
        if (!sortFilter) return;
        
        sortFilter.addEventListener('change', function() {
            currentFilters.sort = this.value;
            applyFilters();
        });
    }
    
    function applyFilters() {
        let filteredItems = [...publicationItems];
        
        // Apply search filter
        if (currentFilters.search) {
            filteredItems = filteredItems.filter(item => {
                const title = item.dataset.title?.toLowerCase() || '';
                const content = item.textContent.toLowerCase();
                return title.includes(currentFilters.search) || 
                       content.includes(currentFilters.search);
            });
        }
        
        // Apply type filter
        if (currentFilters.type !== 'all') {
            filteredItems = filteredItems.filter(item => {
                return item.dataset.type === currentFilters.type;
            });
        }
        
        // Apply year filter
        if (currentFilters.year !== 'all') {
            filteredItems = filteredItems.filter(item => {
                return item.dataset.year === currentFilters.year;
            });
        }
        
        // Apply sorting
        filteredItems.sort((a, b) => {
            switch (currentFilters.sort) {
                case 'date-desc':
                    return new Date(b.dataset.date) - new Date(a.dataset.date);
                case 'date-asc':
                    return new Date(a.dataset.date) - new Date(b.dataset.date);
                case 'title-asc':
                    return (a.dataset.title || '').localeCompare(b.dataset.title || '');
                case 'title-desc':
                    return (b.dataset.title || '').localeCompare(a.dataset.title || '');
                default:
                    return 0;
            }
        });
        
        // Update display
        updatePublicationDisplay(filteredItems);
        updateResultsCount(filteredItems.length);
        
        // Update URL parameters
        updateURLParameters();
    }
    
    function updatePublicationDisplay(filteredItems) {
        // Hide all items first
        publicationItems.forEach(item => {
            item.style.display = 'none';
            item.classList.remove('fade-in');
        });
        
        // Show and animate filtered items
        filteredItems.forEach((item, index) => {
            item.style.display = 'block';
            item.style.order = index;
            
            // Add fade-in animation with staggered delay
            setTimeout(() => {
                item.classList.add('fade-in');
            }, index * 50);
        });
        
        // Show empty state if no results
        showEmptyState(filteredItems.length === 0);
    }
    
    function updateResultsCount(count) {
        if (!resultsCount) return;
        
        const totalCount = publicationItems.length;
        const text = count === totalCount 
            ? `Showing all ${count} publication${count !== 1 ? 's' : ''}`
            : `Showing ${count} of ${totalCount} publication${totalCount !== 1 ? 's' : ''}`;
        
        resultsCount.textContent = text;
    }
    
    function showEmptyState(show) {
        let emptyState = document.querySelector('.empty-state-filtered');
        
        if (show && !emptyState) {
            emptyState = document.createElement('div');
            emptyState.className = 'empty-state-filtered';
            emptyState.innerHTML = `
                <div class="empty-icon">
                    <i data-feather="search"></i>
                </div>
                <h3>No publications found</h3>
                <p>Try adjusting your search terms or filters to find what you're looking for.</p>
                <button class="btn btn-outline" onclick="clearAllFilters()">Clear Filters</button>
            `;
            
            const container = document.querySelector('.publications-list');
            if (container) {
                container.appendChild(emptyState);
                // Re-initialize feather icons
                if (typeof feather !== 'undefined') {
                    feather.replace();
                }
            }
        } else if (!show && emptyState) {
            emptyState.remove();
        }
    }
    
    function clearAllFilters() {
        // Reset filter state
        currentFilters = {
            search: '',
            type: 'all',
            year: 'all',
            sort: 'date-desc'
        };
        
        // Reset form elements
        if (searchInput) searchInput.value = '';
        if (typeFilter) typeFilter.value = 'all';
        if (yearFilter) yearFilter.value = 'all';
        if (sortFilter) sortFilter.value = 'date-desc';
        
        // Apply filters
        applyFilters();
    }
    
    function updateURLParameters() {
        const url = new URL(window.location);
        const params = url.searchParams;
        
        // Update search params
        if (currentFilters.search) {
            params.set('search', currentFilters.search);
        } else {
            params.delete('search');
        }
        
        if (currentFilters.type !== 'all') {
            params.set('type', currentFilters.type);
        } else {
            params.delete('type');
        }
        
        if (currentFilters.year !== 'all') {
            params.set('year', currentFilters.year);
        } else {
            params.delete('year');
        }
        
        if (currentFilters.sort !== 'date-desc') {
            params.set('sort', currentFilters.sort);
        } else {
            params.delete('sort');
        }
        
        // Update URL without page reload
        window.history.replaceState({}, '', url.toString());
    }
    
    function loadURLParameters() {
        const params = new URLSearchParams(window.location.search);
        
        // Load filters from URL
        currentFilters.search = params.get('search') || '';
        currentFilters.type = params.get('type') || 'all';
        currentFilters.year = params.get('year') || 'all';
        currentFilters.sort = params.get('sort') || 'date-desc';
        
        // Update form elements
        if (searchInput) searchInput.value = currentFilters.search;
        if (typeFilter) typeFilter.value = currentFilters.type;
        if (yearFilter) yearFilter.value = currentFilters.year;
        if (sortFilter) sortFilter.value = currentFilters.sort;
    }
    
    function initializeAnimations() {
        // Add CSS for animations if not already present
        if (!document.querySelector('#publications-animations')) {
            const style = document.createElement('style');
            style.id = 'publications-animations';
            style.textContent = `
                .publication-item {
                    opacity: 0;
                    transform: translateY(20px);
                    transition: opacity 0.3s ease, transform 0.3s ease;
                }
                
                .publication-item.fade-in {
                    opacity: 1;
                    transform: translateY(0);
                }
                
                .empty-state-filtered {
                    text-align: center;
                    padding: 4rem 2rem;
                    color: var(--text-secondary);
                }
                
                .empty-state-filtered .empty-icon {
                    width: 64px;
                    height: 64px;
                    margin: 0 auto 1rem;
                    color: var(--text-tertiary);
                }
                
                .empty-state-filtered .empty-icon svg {
                    width: 100%;
                    height: 100%;
                }
                
                .empty-state-filtered h3 {
                    margin-bottom: 0.5rem;
                    color: var(--text-primary);
                }
                
                .empty-state-filtered p {
                    margin-bottom: 1.5rem;
                    max-width: 400px;
                    margin-left: auto;
                    margin-right: auto;
                }
            `;
            document.head.appendChild(style);
        }
    }
    
    function initializeIntersectionObserver() {
        // Lazy loading and animation for publication items
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '50px'
        });
        
        // Observe all publication items
        publicationItems.forEach(item => {
            observer.observe(item);
        });
    }
    
    // Advanced search functionality
    function initializeAdvancedSearch() {
        // Tag-based filtering
        const tagElements = document.querySelectorAll('.tag');
        tagElements.forEach(tag => {
            tag.addEventListener('click', function() {
                const tagText = this.textContent.trim();
                if (searchInput) {
                    searchInput.value = tagText;
                    currentFilters.search = tagText.toLowerCase();
                    applyFilters();
                }
            });
        });
    }
    
    // Export functions for global access
    window.clearAllFilters = clearAllFilters;
    
    // Load URL parameters on page load
    document.addEventListener('DOMContentLoaded', function() {
        loadURLParameters();
        initializeAdvancedSearch();
    });
    
    // Handle browser back/forward buttons
    window.addEventListener('popstate', function() {
        loadURLParameters();
        applyFilters();
    });
    
})();

// Search functionality for the global search overlay
(function() {
    'use strict';
    
    let searchData = [];
    let searchIndex = null;
    
    // Initialize search when DOM is loaded
    document.addEventListener('DOMContentLoaded', function() {
        initializeGlobalSearch();
    });
    
    function initializeGlobalSearch() {
        const searchInput = document.getElementById('search-input');
        const searchResults = document.getElementById('search-results');
        
        if (!searchInput || !searchResults) return;
        
        // Build search index
        buildSearchIndex();
        
        // Real-time search
        let searchTimeout;
        searchInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            const query = this.value.trim();
            
            if (query.length < 2) {
                searchResults.innerHTML = '';
                return;
            }
            
            searchTimeout = setTimeout(() => {
                performSearch(query, searchResults);
            }, 200);
        });
    }
    
    function buildSearchIndex() {
        // Collect all searchable content
        const publications = document.querySelectorAll('.publication-item');
        
        publications.forEach(pub => {
            const title = pub.dataset.title || '';
            const type = pub.dataset.type || '';
            const year = pub.dataset.year || '';
            const content = pub.textContent || '';
            const url = pub.querySelector('a')?.href || '';
            
            searchData.push({
                title,
                type,
                year,
                content: content.toLowerCase(),
                url,
                element: pub
            });
        });
    }
    
    function performSearch(query, resultsContainer) {
        const queryLower = query.toLowerCase();
        const results = [];
        
        searchData.forEach(item => {
            let score = 0;
            
            // Title match (highest priority)
            if (item.title.toLowerCase().includes(queryLower)) {
                score += 10;
            }
            
            // Type match
            if (item.type.toLowerCase().includes(queryLower)) {
                score += 5;
            }
            
            // Content match
            if (item.content.includes(queryLower)) {
                score += 1;
            }
            
            if (score > 0) {
                results.push({ ...item, score });
            }
        });
        
        // Sort by relevance
        results.sort((a, b) => b.score - a.score);
        
        // Display results
        displaySearchResults(results.slice(0, 5), resultsContainer, query);
    }
    
    function displaySearchResults(results, container, query) {
        if (results.length === 0) {
            container.innerHTML = `
                <div class="search-no-results">
                    <p>No results found for "${escapeHtml(query)}"</p>
                </div>
            `;
            return;
        }
        
        const resultsHTML = results.map(result => `
            <div class="search-result-item">
                <h4><a href="${result.url}">${highlightText(result.title, query)}</a></h4>
                <div class="search-result-meta">
                    <span class="result-type">${result.type}</span>
                    <span class="result-year">${result.year}</span>
                </div>
            </div>
        `).join('');
        
        container.innerHTML = resultsHTML;
    }
    
    function highlightText(text, query) {
        const regex = new RegExp(`(${escapeRegex(query)})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }
    
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    function escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
    
})();

// Smooth scrolling and other enhancements
(function() {
    'use strict';
    
    document.addEventListener('DOMContentLoaded', function() {
        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
        
        // Add loading states for dynamic content
        const dynamicElements = document.querySelectorAll('[data-dynamic]');
        dynamicElements.forEach(element => {
            element.classList.add('loading');
            
            // Simulate loading completion
            setTimeout(() => {
                element.classList.remove('loading');
                element.classList.add('loaded');
            }, Math.random() * 1000 + 500);
        });
    });
    
})();