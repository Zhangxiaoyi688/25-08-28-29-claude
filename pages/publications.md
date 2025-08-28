---
layout: page
title: "Publications"
subtitle: "Research Papers and Academic Contributions"
description: "Explore my research publications in machine learning, natural language processing, and artificial intelligence"
permalink: /publications/
custom_js: publications.js
---

<!-- Publications Filter Navigation is included via navigation.html -->

<div class="publications-container">
    <div class="publications-stats">
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-number">{{ site.research | size }}</div>
                <div class="stat-label">Total Publications</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{{ site.research | where: 'type', 'conference' | size }}</div>
                <div class="stat-label">Conference Papers</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{{ site.research | where: 'type', 'journal' | size }}</div>
                <div class="stat-label">Journal Articles</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{{ site.research | map: 'date' | map: 'year' | uniq | size }}</div>
                <div class="stat-label">Years Active</div>
            </div>
        </div>
    </div>
    
    <div class="publications-list">
        {% assign sorted_papers = site.research | sort: 'date' | reverse %}
        {% for paper in sorted_papers %}
            <article class="publication-item" 
                     data-type="{{ paper.type }}" 
                     data-year="{{ paper.date | date: '%Y' }}" 
                     data-date="{{ paper.date | date: '%Y-%m-%d' }}" 
                     data-title="{{ paper.title }}">
                <div class="publication-content">
                    <div class="publication-header">
                        <div class="publication-meta">
                            {% assign paper_type = site.research_types | where: 'type', paper.type | first %}
                            <span class="type-badge" style="background-color: {{ paper_type.color }}">
                                {{ paper_type.name | default: paper.type | capitalize }}
                            </span>
                            <span class="publication-year">{{ paper.date | date: "%Y" }}</span>
                        </div>
                        
                        <div class="publication-actions-inline">
                            {% if paper.pdf_url %}
                                <a href="{{ paper.pdf_url | relative_url }}" 
                                   class="action-link" 
                                   target="_blank" 
                                   title="Download PDF">
                                    <i data-feather="download"></i>
                                </a>
                            {% endif %}
                            {% if paper.external_url %}
                                <a href="{{ paper.external_url }}" 
                                   class="action-link" 
                                   target="_blank" 
                                   title="View on Publisher Site">
                                    <i data-feather="external-link"></i>
                                </a>
                            {% endif %}
                            <button class="action-link" 
                                    onclick="copyBibTeX('{{ paper.title | slugify }}')"
                                    title="Copy BibTeX">
                                <i data-feather="copy"></i>
                            </button>
                        </div>
                    </div>
                    
                    <h3 class="publication-title">
                        <a href="{{ paper.url | relative_url }}">{{ paper.title }}</a>
                    </h3>
                    
                    {% if paper.authors %}
                        <div class="publication-authors">
                            <i data-feather="users"></i>
                            <span>{{ paper.authors | join: ", " }}</span>
                        </div>
                    {% endif %}
                    
                    {% if paper.venue %}
                        <div class="publication-venue">
                            <i data-feather="map-pin"></i>
                            <span>{{ paper.venue }}</span>
                        </div>
                    {% endif %}
                    
                    {% if paper.abstract %}
                        <div class="publication-abstract">
                            <p>{{ paper.abstract | truncate: 200 }}</p>
                        </div>
                    {% endif %}
                    
                    {% if paper.tags %}
                        <div class="publication-tags">
                            {% for tag in paper.tags limit: 5 %}
                                <span class="tag">{{ tag }}</span>
                            {% endfor %}
                        </div>
                    {% endif %}
                    
                    <div class="publication-footer">
                        <div class="publication-date">
                            <i data-feather="calendar"></i>
                            <span>{{ paper.date | date: "%B %d, %Y" }}</span>
                        </div>
                        
                        <a href="{{ paper.url | relative_url }}" class="read-more-link">
                            Read More <i data-feather="arrow-right"></i>
                        </a>
                    </div>
                </div>
                
                <!-- Hidden BibTeX for copying -->
                <div class="bibtex-hidden" id="bibtex-{{ paper.title | slugify }}" style="display: none;">
@{{ paper.type | default: 'article' }}{{"{" }}{{ paper.title | slugify }},
  title={{"{" }}{{ paper.title }}{"}" }},
  author={{"{" }}{{ paper.authors | join: " and " }}{"}" }},
  {% if paper.venue %}venue={{"{" }}{{ paper.venue }}{"}" }},{% endif %}
  year={{"{" }}{{ paper.date | date: "%Y" }}{"}" }}
}
                </div>
            </article>
        {% endfor %}
    </div>
    
    {% if site.research.size == 0 %}
        <div class="empty-state">
            <div class="empty-icon">
                <i data-feather="file-text"></i>
            </div>
            <h3>No Publications Yet</h3>
            <p>Publications will appear here once research papers are added to the _research folder.</p>
        </div>
    {% endif %}
</div>

<style>
.publications-container {
    max-width: 1000px;
    margin: 0 auto;
}

.publications-stats {
    margin-bottom: var(--spacing-12);
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: var(--spacing-6);
}

.stat-item {
    text-align: center;
    padding: var(--spacing-6);
    background-color: var(--bg-secondary);
    border-radius: var(--radius-xl);
    border: 1px solid var(--border-primary);
}

.stat-number {
    font-size: var(--font-size-3xl);
    font-weight: var(--font-weight-bold);
    color: var(--color-primary);
    margin-bottom: var(--spacing-2);
}

.stat-label {
    font-size: var(--font-size-sm);
    color: var(--text-secondary);
    font-weight: var(--font-weight-medium);
}

.publication-item {
    background-color: var(--bg-primary);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-xl);
    padding: var(--spacing-6);
    margin-bottom: var(--spacing-6);
    transition: all var(--transition-normal);
}

.publication-item:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.publication-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: var(--spacing-4);
}

.publication-meta {
    display: flex;
    align-items: center;
    gap: var(--spacing-3);
}

.publication-year {
    font-size: var(--font-size-sm);
    color: var(--text-tertiary);
    font-weight: var(--font-weight-medium);
}

.publication-actions-inline {
    display: flex;
    gap: var(--spacing-2);
}

.action-link {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: var(--radius-lg);
    background-color: var(--bg-secondary);
    color: var(--text-secondary);
    text-decoration: none;
    border: none;
    cursor: pointer;
    transition: all var(--transition-fast);
}

.action-link:hover {
    background-color: var(--color-primary);
    color: var(--color-white);
}

.publication-title {
    font-size: var(--font-size-xl);
    font-weight: var(--font-weight-semibold);
    line-height: var(--line-height-tight);
    margin-bottom: var(--spacing-3);
}

.publication-title a {
    color: var(--text-primary);
    text-decoration: none;
}

.publication-title a:hover {
    color: var(--color-primary);
}

.publication-authors,
.publication-venue {
    display: flex;
    align-items: center;
    gap: var(--spacing-2);
    font-size: var(--font-size-sm);
    color: var(--text-secondary);
    margin-bottom: var(--spacing-2);
}

.publication-authors svg,
.publication-venue svg {
    width: 14px;
    height: 14px;
    color: var(--color-primary);
}

.publication-abstract {
    margin: var(--spacing-4) 0;
}

.publication-abstract p {
    font-size: var(--font-size-sm);
    color: var(--text-secondary);
    line-height: var(--line-height-relaxed);
    margin-bottom: 0;
}

.publication-tags {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-2);
    margin: var(--spacing-4) 0;
}

.publication-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: var(--spacing-4);
    padding-top: var(--spacing-4);
    border-top: 1px solid var(--border-primary);
}

.publication-date {
    display: flex;
    align-items: center;
    gap: var(--spacing-1);
    font-size: var(--font-size-xs);
    color: var(--text-tertiary);
}

.publication-date svg {
    width: 12px;
    height: 12px;
}

.read-more-link {
    display: flex;
    align-items: center;
    gap: var(--spacing-1);
    font-size: var(--font-size-sm);
    font-weight: var(--font-weight-medium);
    color: var(--color-primary);
    text-decoration: none;
    transition: color var(--transition-fast);
}

.read-more-link:hover {
    color: var(--color-primary-light);
}

.read-more-link svg {
    width: 14px;
    height: 14px;
    transition: transform var(--transition-fast);
}

.read-more-link:hover svg {
    transform: translateX(2px);
}

.empty-state {
    text-align: center;
    padding: var(--spacing-16) var(--spacing-4);
    color: var(--text-secondary);
}

.empty-icon {
    width: 64px;
    height: 64px;
    margin: 0 auto var(--spacing-4);
    color: var(--text-tertiary);
}

.empty-icon svg {
    width: 100%;
    height: 100%;
}

@media (max-width: 767px) {
    .publication-header {
        flex-direction: column;
        gap: var(--spacing-3);
    }
    
    .publication-footer {
        flex-direction: column;
        gap: var(--spacing-3);
        align-items: flex-start;
    }
}
</style>

<script>
function copyBibTeX(paperId) {
    const bibTexElement = document.getElementById('bibtex-' + paperId);
    if (bibTexElement) {
        const bibTexContent = bibTexElement.textContent.trim();
        navigator.clipboard.writeText(bibTexContent).then(function() {
            // Show success feedback
            const button = event.target.closest('button');
            const icon = button.querySelector('i');
            const originalFeather = icon.getAttribute('data-feather');
            
            icon.setAttribute('data-feather', 'check');
            feather.replace();
            
            setTimeout(() => {
                icon.setAttribute('data-feather', originalFeather);
                feather.replace();
            }, 2000);
        }).catch(function(err) {
            console.error('Could not copy BibTeX: ', err);
        });
    }
}
</script>