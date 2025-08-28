---
layout: page
title: "About Me"
subtitle: "Academic Background and Research Journey"
description: "Learn more about my academic background, research interests, and professional experience in artificial intelligence and machine learning"
permalink: /about/
---

<div class="about-container">
    <div class="about-intro">
        <div class="intro-content">
            <div class="intro-text">
                <h2>{{ site.researcher.name }}</h2>
                <p class="intro-title">{{ site.researcher.title }}</p>
                <p class="intro-institution">{{ site.researcher.institution }}</p>
                <p class="intro-bio">{{ site.researcher.bio }}</p>
                
                <div class="contact-info">
                    <div class="contact-item">
                        <i data-feather="mail"></i>
                        <a href="mailto:{{ site.researcher.email }}">{{ site.researcher.email }}</a>
                    </div>
                    <div class="contact-item">
                        <i data-feather="map-pin"></i>
                        <span>{{ site.researcher.institution }}</span>
                    </div>
                </div>
                
                <div class="social-links">
                    {% for link in site.researcher.social_links %}
                        <a href="{{ link.url }}" target="_blank" rel="noopener" class="social-link" title="{{ link.name }}">
                            <i data-feather="{{ link.icon }}"></i>
                            <span>{{ link.name }}</span>
                        </a>
                    {% endfor %}
                </div>
            </div>
            
            <div class="intro-image">
                <img src="{{ site.researcher.profile_image | relative_url }}" alt="{{ site.researcher.name }}" class="profile-image">
            </div>
        </div>
    </div>
    
    <div class="about-sections">
        <section class="about-section">
            <h3><i data-feather="graduation-cap"></i> Education</h3>
            <div class="timeline">
                <div class="timeline-item">
                    <div class="timeline-marker"></div>
                    <div class="timeline-content">
                        <h4>Ph.D. in Computer Science</h4>
                        <p class="timeline-institution">Stanford University</p>
                        <p class="timeline-date">2018 - 2022</p>
                        <p class="timeline-description">Dissertation: "Advanced Neural Architectures for Natural Language Understanding"</p>
                        <p class="timeline-advisor">Advisor: Prof. Christopher Manning</p>
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-marker"></div>
                    <div class="timeline-content">
                        <h4>M.S. in Machine Learning</h4>
                        <p class="timeline-institution">Carnegie Mellon University</p>
                        <p class="timeline-date">2016 - 2018</p>
                        <p class="timeline-description">Focus on deep learning and computer vision applications</p>
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-marker"></div>
                    <div class="timeline-content">
                        <h4>B.S. in Computer Science</h4>
                        <p class="timeline-institution">MIT</p>
                        <p class="timeline-date">2012 - 2016</p>
                        <p class="timeline-description">Magna Cum Laude, Phi Beta Kappa</p>
                    </div>
                </div>
            </div>
        </section>
        
        <section class="about-section">
            <h3><i data-feather="briefcase"></i> Professional Experience</h3>
            <div class="experience-grid">
                <div class="experience-item">
                    <div class="experience-header">
                        <h4>Assistant Professor</h4>
                        <span class="experience-date">2022 - Present</span>
                    </div>
                    <p class="experience-company">University of Technology</p>
                    <p class="experience-description">
                        Leading research in natural language processing and machine learning. 
                        Teaching graduate and undergraduate courses in AI and deep learning.
                    </p>
                    <ul class="experience-achievements">
                        <li>Published 15+ papers in top-tier conferences (NeurIPS, ICML, ACL)</li>
                        <li>Received NSF CAREER Award for research in neural language models</li>
                        <li>Supervising 8 PhD students and 12 master's students</li>
                    </ul>
                </div>
                
                <div class="experience-item">
                    <div class="experience-header">
                        <h4>Research Scientist Intern</h4>
                        <span class="experience-date">Summer 2021</span>
                    </div>
                    <p class="experience-company">Google Research</p>
                    <p class="experience-description">
                        Worked on large-scale language models and their applications to code generation.
                    </p>
                    <ul class="experience-achievements">
                        <li>Developed novel attention mechanisms for code understanding</li>
                        <li>Contributed to internal tools used by thousands of developers</li>
                    </ul>
                </div>
                
                <div class="experience-item">
                    <div class="experience-header">
                        <h4>Research Assistant</h4>
                        <span class="experience-date">2018 - 2022</span>
                    </div>
                    <p class="experience-company">Stanford NLP Group</p>
                    <p class="experience-description">
                        Conducted research on transformer architectures and their applications to various NLP tasks.
                    </p>
                    <ul class="experience-achievements">
                        <li>Co-authored 8 papers published in ACL, EMNLP, and NAACL</li>
                        <li>Developed open-source tools with 1000+ GitHub stars</li>
                    </ul>
                </div>
            </div>
        </section>
        
        <section class="about-section">
            <h3><i data-feather="award"></i> Awards & Honors</h3>
            <div class="awards-grid">
                <div class="award-item">
                    <div class="award-icon">
                        <i data-feather="trophy"></i>
                    </div>
                    <div class="award-content">
                        <h4>NSF CAREER Award</h4>
                        <p class="award-year">2023</p>
                        <p class="award-description">For outstanding research in neural language models</p>
                    </div>
                </div>
                
                <div class="award-item">
                    <div class="award-icon">
                        <i data-feather="star"></i>
                    </div>
                    <div class="award-content">
                        <h4>Best Paper Award</h4>
                        <p class="award-year">2022</p>
                        <p class="award-description">ACL 2022 - "Efficient Transformers for Long Sequences"</p>
                    </div>
                </div>
                
                <div class="award-item">
                    <div class="award-icon">
                        <i data-feather="users"></i>
                    </div>
                    <div class="award-content">
                        <h4>Outstanding Reviewer</h4>
                        <p class="award-year">2021</p>
                        <p class="award-description">NeurIPS 2021 - Top 10% of reviewers</p>
                    </div>
                </div>
                
                <div class="award-item">
                    <div class="award-icon">
                        <i data-feather="book"></i>
                    </div>
                    <div class="award-content">
                        <h4>Stanford Graduate Fellowship</h4>
                        <p class="award-year">2018-2022</p>
                        <p class="award-description">Full funding for PhD studies</p>
                    </div>
                </div>
            </div>
        </section>
        
        <section class="about-section">
            <h3><i data-feather="target"></i> Research Interests</h3>
            <div class="research-interests">
                <div class="interest-category">
                    <h4>Natural Language Processing</h4>
                    <ul>
                        <li>Large Language Models and Transformers</li>
                        <li>Text Generation and Summarization</li>
                        <li>Question Answering Systems</li>
                        <li>Multilingual NLP</li>
                    </ul>
                </div>
                
                <div class="interest-category">
                    <h4>Machine Learning</h4>
                    <ul>
                        <li>Deep Learning Architectures</li>
                        <li>Optimization and Training Techniques</li>
                        <li>Few-shot and Zero-shot Learning</li>
                        <li>Interpretability and Explainability</li>
                    </ul>
                </div>
                
                <div class="interest-category">
                    <h4>AI Applications</h4>
                    <ul>
                        <li>Code Generation and Programming Assistance</li>
                        <li>Scientific Text Analysis</li>
                        <li>Educational Technology</li>
                        <li>AI Safety and Alignment</li>
                    </ul>
                </div>
            </div>
        </section>
        
        <section class="about-section">
            <h3><i data-feather="users"></i> Service & Leadership</h3>
            <div class="service-content">
                <div class="service-category">
                    <h4>Editorial & Review</h4>
                    <ul>
                        <li>Associate Editor: Transactions of the Association for Computational Linguistics (2023-present)</li>
                        <li>Program Committee: NeurIPS, ICML, ACL, EMNLP, NAACL (2020-present)</li>
                        <li>Senior Program Committee: AAAI (2022-present)</li>
                        <li>Reviewer: Nature Machine Intelligence, JMLR</li>
                    </ul>
                </div>
                
                <div class="service-category">
                    <h4>Community & Outreach</h4>
                    <ul>
                        <li>Co-organizer: Workshop on Efficient NLP at EMNLP 2023</li>
                        <li>Mentor: AI4ALL summer program (2021-present)</li>
                        <li>Speaker: Various industry talks and academic seminars</li>
                        <li>Volunteer: Local STEM education initiatives</li>
                    </ul>
                </div>
            </div>
        </section>
    </div>
</div>

<style>
.about-container {
    max-width: 1000px;
    margin: 0 auto;
}

.about-intro {
    margin-bottom: var(--spacing-16);
}

.intro-content {
    display: grid;
    grid-template-columns: 1fr 300px;
    gap: var(--spacing-12);
    align-items: start;
}

@media (max-width: 767px) {
    .intro-content {
        grid-template-columns: 1fr;
        gap: var(--spacing-8);
        text-align: center;
    }
}

.intro-text h2 {
    font-size: var(--font-size-3xl);
    font-weight: var(--font-weight-bold);
    color: var(--text-primary);
    margin-bottom: var(--spacing-2);
}

.intro-title {
    font-size: var(--font-size-xl);
    font-weight: var(--font-weight-semibold);
    color: var(--color-primary);
    margin-bottom: var(--spacing-1);
}

.intro-institution {
    font-size: var(--font-size-lg);
    color: var(--text-secondary);
    margin-bottom: var(--spacing-4);
}

.intro-bio {
    font-size: var(--font-size-base);
    line-height: var(--line-height-relaxed);
    color: var(--text-secondary);
    margin-bottom: var(--spacing-6);
}

.contact-info {
    margin-bottom: var(--spacing-6);
}

.contact-item {
    display: flex;
    align-items: center;
    gap: var(--spacing-2);
    margin-bottom: var(--spacing-3);
    font-size: var(--font-size-sm);
    color: var(--text-secondary);
}

.contact-item svg {
    width: 16px;
    height: 16px;
    color: var(--color-primary);
}

.contact-item a {
    color: inherit;
    text-decoration: none;
}

.contact-item a:hover {
    color: var(--color-primary);
}

.social-links {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-3);
}

.social-link {
    display: flex;
    align-items: center;
    gap: var(--spacing-2);
    padding: var(--spacing-2) var(--spacing-4);
    background-color: var(--bg-secondary);
    color: var(--text-secondary);
    text-decoration: none;
    border-radius: var(--radius-lg);
    font-size: var(--font-size-sm);
    transition: all var(--transition-fast);
}

.social-link:hover {
    background-color: var(--color-primary);
    color: var(--color-white);
}

.social-link svg {
    width: 16px;
    height: 16px;
}

.profile-image {
    width: 100%;
    height: auto;
    border-radius: var(--radius-2xl);
    box-shadow: var(--shadow-lg);
}

.about-sections {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-16);
}

.about-section h3 {
    display: flex;
    align-items: center;
    gap: var(--spacing-2);
    font-size: var(--font-size-2xl);
    font-weight: var(--font-weight-semibold);
    color: var(--text-primary);
    margin-bottom: var(--spacing-8);
    padding-bottom: var(--spacing-3);
    border-bottom: 2px solid var(--color-primary);
}

.about-section h3 svg {
    width: 24px;
    height: 24px;
    color: var(--color-primary);
}

/* Timeline Styles */
.timeline {
    position: relative;
    padding-left: var(--spacing-8);
}

.timeline::before {
    content: '';
    position: absolute;
    left: 12px;
    top: 0;
    bottom: 0;
    width: 2px;
    background-color: var(--border-primary);
}

.timeline-item {
    position: relative;
    margin-bottom: var(--spacing-8);
}

.timeline-marker {
    position: absolute;
    left: -20px;
    top: 8px;
    width: 12px;
    height: 12px;
    background-color: var(--color-primary);
    border-radius: var(--radius-full);
    border: 3px solid var(--bg-primary);
    box-shadow: 0 0 0 2px var(--color-primary);
}

.timeline-content h4 {
    font-size: var(--font-size-lg);
    font-weight: var(--font-weight-semibold);
    color: var(--text-primary);
    margin-bottom: var(--spacing-2);
}

.timeline-institution {
    font-size: var(--font-size-base);
    font-weight: var(--font-weight-medium);
    color: var(--color-primary);
    margin-bottom: var(--spacing-1);
}

.timeline-date {
    font-size: var(--font-size-sm);
    color: var(--text-tertiary);
    margin-bottom: var(--spacing-3);
}

.timeline-description,
.timeline-advisor {
    font-size: var(--font-size-sm);
    color: var(--text-secondary);
    line-height: var(--line-height-relaxed);
    margin-bottom: var(--spacing-2);
}

/* Experience Styles */
.experience-grid {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-8);
}

.experience-item {
    padding: var(--spacing-6);
    background-color: var(--bg-secondary);
    border-radius: var(--radius-xl);
    border: 1px solid var(--border-primary);
}

.experience-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: var(--spacing-2);
}

.experience-header h4 {
    font-size: var(--font-size-lg);
    font-weight: var(--font-weight-semibold);
    color: var(--text-primary);
    margin-bottom: 0;
}

.experience-date {
    font-size: var(--font-size-sm);
    color: var(--text-tertiary);
    font-weight: var(--font-weight-medium);
}

.experience-company {
    font-size: var(--font-size-base);
    font-weight: var(--font-weight-medium);
    color: var(--color-primary);
    margin-bottom: var(--spacing-3);
}

.experience-description {
    font-size: var(--font-size-sm);
    color: var(--text-secondary);
    line-height: var(--line-height-relaxed);
    margin-bottom: var(--spacing-4);
}

.experience-achievements {
    list-style: none;
    padding: 0;
    margin: 0;
}

.experience-achievements li {
    position: relative;
    padding-left: var(--spacing-4);
    font-size: var(--font-size-sm);
    color: var(--text-secondary);
    line-height: var(--line-height-relaxed);
    margin-bottom: var(--spacing-2);
}

.experience-achievements li::before {
    content: '•';
    position: absolute;
    left: 0;
    color: var(--color-primary);
    font-weight: var(--font-weight-bold);
}

/* Awards Styles */
.awards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: var(--spacing-6);
}

.award-item {
    display: flex;
    gap: var(--spacing-4);
    padding: var(--spacing-6);
    background-color: var(--bg-secondary);
    border-radius: var(--radius-xl);
    border: 1px solid var(--border-primary);
}

.award-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    background-color: var(--color-primary);
    color: var(--color-white);
    border-radius: var(--radius-full);
    flex-shrink: 0;
}

.award-icon svg {
    width: 24px;
    height: 24px;
}

.award-content h4 {
    font-size: var(--font-size-base);
    font-weight: var(--font-weight-semibold);
    color: var(--text-primary);
    margin-bottom: var(--spacing-1);
}

.award-year {
    font-size: var(--font-size-sm);
    color: var(--color-primary);
    font-weight: var(--font-weight-medium);
    margin-bottom: var(--spacing-2);
}

.award-description {
    font-size: var(--font-size-sm);
    color: var(--text-secondary);
    line-height: var(--line-height-relaxed);
    margin-bottom: 0;
}

/* Research Interests Styles */
.research-interests {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: var(--spacing-8);
}

.interest-category {
    padding: var(--spacing-6);
    background-color: var(--bg-secondary);
    border-radius: var(--radius-xl);
    border: 1px solid var(--border-primary);
}

.interest-category h4 {
    font-size: var(--font-size-lg);
    font-weight: var(--font-weight-semibold);
    color: var(--text-primary);
    margin-bottom: var(--spacing-4);
}

.interest-category ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.interest-category li {
    position: relative;
    padding-left: var(--spacing-4);
    font-size: var(--font-size-sm);
    color: var(--text-secondary);
    line-height: var(--line-height-relaxed);
    margin-bottom: var(--spacing-2);
}

.interest-category li::before {
    content: '▸';
    position: absolute;
    left: 0;
    color: var(--color-primary);
    font-weight: var(--font-weight-bold);
}

/* Service Styles */
.service-content {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: var(--spacing-8);
}

.service-category {
    padding: var(--spacing-6);
    background-color: var(--bg-secondary);
    border-radius: var(--radius-xl);
    border: 1px solid var(--border-primary);
}

.service-category h4 {
    font-size: var(--font-size-lg);
    font-weight: var(--font-weight-semibold);
    color: var(--text-primary);
    margin-bottom: var(--spacing-4);
}

.service-category ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.service-category li {
    position: relative;
    padding-left: var(--spacing-4);
    font-size: var(--font-size-sm);
    color: var(--text-secondary);
    line-height: var(--line-height-relaxed);
    margin-bottom: var(--spacing-3);
}

.service-category li::before {
    content: '•';
    position: absolute;
    left: 0;
    color: var(--color-primary);
    font-weight: var(--font-weight-bold);
}

@media (max-width: 767px) {
    .awards-grid,
    .research-interests,
    .service-content {
        grid-template-columns: 1fr;
    }
    
    .experience-header {
        flex-direction: column;
        gap: var(--spacing-1);
    }
    
    .award-item {
        flex-direction: column;
        text-align: center;
    }
}
</style>